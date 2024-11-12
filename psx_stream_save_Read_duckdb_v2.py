import asyncio
import websockets
import duckdb
import pandas as pd
import json
import base64
import os
import time
from datetime import datetime, timezone
from colorama import init, Fore
from asyncio import Lock

# Initialize Colorama for colored terminal output
init(autoreset=True)

# Global control for enabling/disabling debug logs
DEBUG_MODE = False  # Set to False to disable debug logs

# Function to print debug logs based on the DEBUG_MODE
def debug_log(message, color=Fore.CYAN):
    if DEBUG_MODE:
        print(color + message + Fore.RESET)

# Configuration Options
CONFIGURATIONS = {
    "ultra_low": {
        "DUCKDB_FILE_PATH": 'stock_data_ultra_low.duckdb',
        "FLUSH_INTERVAL": 120,
        "MAX_BUFFER_SIZE": 10000,
        "CHUNK_SIZE": 2000
    },
    "low": {
        "DUCKDB_FILE_PATH": 'stock_data_low.duckdb',
        "FLUSH_INTERVAL": 60,
        "MAX_BUFFER_SIZE": 5000,
        "CHUNK_SIZE": 1000
    },
    "medium": {
        "DUCKDB_FILE_PATH": 'stock_data_medium.duckdb',
        "FLUSH_INTERVAL": 30,
        "MAX_BUFFER_SIZE": 3000,
        "CHUNK_SIZE": 500
    },
    "high": {
        "DUCKDB_FILE_PATH": 'stock_data_high.duckdb',
        "FLUSH_INTERVAL": 10,
        "MAX_BUFFER_SIZE": 1000,
        "CHUNK_SIZE": 200
    },
    "ultra_high": {
        "DUCKDB_FILE_PATH": 'stock_data_ultra_high.duckdb',
        "FLUSH_INTERVAL": 5,
        "MAX_BUFFER_SIZE": 500,
        "CHUNK_SIZE": 100
    },
    "nvme_optimized": {
        "DUCKDB_FILE_PATH": 'stock_data_nvme.duckdb',
        "FLUSH_INTERVAL": 2,
        "MAX_BUFFER_SIZE": 200,
        "CHUNK_SIZE": 50
    },
    "real_time": {
        "DUCKDB_FILE_PATH": 'stock_data_real_time.duckdb',
        "FLUSH_INTERVAL": 1,
        "MAX_BUFFER_SIZE": 100,
        "CHUNK_SIZE": 20
    }
}

# Select your configuration here
CONFIG_CHOICE = "high"  # Change this to "ultra_low", "low", "medium", etc.
CONFIG = CONFIGURATIONS[CONFIG_CHOICE]

# Configuration Variables
DUCKDB_FILE_PATH = CONFIG["DUCKDB_FILE_PATH"]
FLUSH_INTERVAL = CONFIG["FLUSH_INTERVAL"]
MAX_BUFFER_SIZE = CONFIG["MAX_BUFFER_SIZE"]
CHUNK_SIZE = CONFIG["CHUNK_SIZE"]

# In-memory buffer for storing data temporarily
data_buffer = []
buffer_lock = Lock()

# Global DuckDB setup
def setup_duckdb():
    """Setup the DuckDB database and create the table if it doesn't exist."""
    try:
        with duckdb.connect(DUCKDB_FILE_PATH) as db:
            db.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT,
                    market TEXT,
                    status TEXT,
                    timestamp TIMESTAMP,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    last_close REAL,
                    change REAL,
                    change_percent REAL,
                    bid_price REAL,
                    bid_volume INTEGER,
                    ask_price REAL,
                    ask_volume INTEGER,
                    total_value REAL,
                    transactions INTEGER,
                    lt_time TIMESTAMP,
                    lt_price REAL,
                    lt_volume INTEGER
                )
            ''')
            debug_log(f"Database setup complete for {CONFIG_CHOICE} configuration.", Fore.GREEN)
    except Exception as e:
        debug_log(f"Error setting up DuckDB: {e}", Fore.RED)

setup_duckdb()

def generate_websocket_key():
    random_bytes = os.urandom(16)
    return base64.b64encode(random_bytes).decode('utf-8')

def get_precise_timestamp(seconds):
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    current_millis = int((time.time() % 1) * 1000)
    return dt.replace(microsecond=current_millis * 1000)

async def store_data(data):
    """Store the received data into the in-memory buffer."""
    global data_buffer
    try:
        lt_data = data.get("lt")
        lt_time, lt_price, lt_volume = None, None, None

        if lt_data:
            lt_time = get_precise_timestamp(lt_data['t'])
            lt_price = lt_data.get('x')
            lt_volume = lt_data.get('v')

        timestamp = get_precise_timestamp(data['t'])

        record = (
            data['s'], data['m'], data['st'], timestamp,
            data['o'], data['h'], data['l'], data['c'], data['v'],
            data.get('ldcp'), data.get('ch'), round(data.get('pch', 0) * 100, 2),
            data.get('bp'), data.get('bv'), data.get('ap'), data.get('av'),
            data.get('val'), data.get('tr'), lt_time, lt_price, lt_volume
        )

        async with buffer_lock:
            data_buffer.append(record)
            debug_log(f"Data stored in buffer. Buffer size: {len(data_buffer)}", Fore.GREEN)

    except KeyError as e:
        debug_log(f"KeyError while storing data: {e}. Data: {data}", Fore.RED)
    except TypeError as e:
        debug_log(f"TypeError while storing data: {e}. Data: {data}", Fore.RED)
    except Exception as e:
        debug_log(f"Unexpected error while storing data: {e}. Data: {data}", Fore.RED)

async def flush_data_to_duckdb():
    """Flush the buffered data to the DuckDB database."""
    global data_buffer
    async with buffer_lock:
        if not data_buffer:
            debug_log("No data to flush.", Fore.YELLOW)
            return

        try:
            debug_log(f"Flushing {len(data_buffer)} records to DuckDB...", Fore.CYAN)
            df = pd.DataFrame(data_buffer, columns=[
                "symbol", "market", "status", "timestamp", "open", "high", "low",
                "close", "volume", "last_close", "change", "change_percent",
                "bid_price", "bid_volume", "ask_price", "ask_volume",
                "total_value", "transactions", "lt_time", "lt_price", "lt_volume"
            ])
            with duckdb.connect(DUCKDB_FILE_PATH) as db:
                db.execute("INSERT INTO stock_data SELECT * FROM df")
            data_buffer.clear()
            debug_log("Data successfully flushed to DuckDB.", Fore.GREEN)
        except duckdb.Error as e:
            debug_log(f"DuckDB error during flush: {e}", Fore.RED)
        except Exception as e:
            debug_log(f"Unexpected error during flush: {e}", Fore.RED)

async def periodic_flush():
    """Periodically flush data from the buffer to the database."""
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL)
            await flush_data_to_duckdb()
        except Exception as e:
            debug_log(f"Error in periodic flush: {e}", Fore.RED)

async def connect():
    """Connect to the WebSocket server and handle incoming messages."""
    uri = "wss://market.capitalstake.com/stream"
    headers = {"sec-websocket-key": generate_websocket_key()}

    while True:
        try:
            async with websockets.connect(
                uri,
                extra_headers=headers,
                ping_interval=240,  # Increased from 120 to 240 seconds
                ping_timeout=120    # Increased from 60 to 120 seconds
            ) as websocket:
                debug_log("Connected to WebSocket!", Fore.GREEN)

                # Start the periodic flush task
                asyncio.create_task(periodic_flush())

                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        debug_log(f"Received data: {data}")
                        if data.get("t") == "tick" and "d" in data:
                            await store_data(data["d"])
                        else:
                            debug_log(f"Unexpected data format: {data}", Fore.YELLOW)
                    except websockets.ConnectionClosedError as e:
                        debug_log(f"WebSocket connection closed: {e}. Reconnecting...", Fore.RED)
                        break
                    except json.JSONDecodeError as e:
                        debug_log(f"JSON decode error: {e}. Message: {message}", Fore.RED)
                    except Exception as e:
                        debug_log(f"Unexpected error while receiving data: {e}", Fore.RED)
        except websockets.ConnectionClosedError as e:
            debug_log(f"WebSocket closed: {e}. Reconnecting in 5 seconds...", Fore.RED)
        except Exception as e:
            debug_log(f"Unexpected connection error: {e}. Retrying in 5 seconds...", Fore.RED)
        await asyncio.sleep(5)

def main():
    try:
        asyncio.run(connect())
    except KeyboardInterrupt:
        debug_log("Stream interrupted by user. Exiting...", Fore.YELLOW)
    finally:
        # Flush any remaining data before exiting
        loop = asyncio.get_event_loop()
        loop.run_until_complete(flush_data_to_duckdb())
        debug_log("Database connection closed.", Fore.CYAN)

if __name__ == "__main__":
    main()
