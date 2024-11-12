import asyncio
import websockets
import duckdb
import pandas as pd
import json
import base64
import os
from datetime import datetime
from colorama import init, Fore
from asyncio import Lock

# Initialize Colorama for colored terminal output
init(autoreset=True)

# Global control for enabling/disabling debug logs
DEBUG_MODE = False

def debug_log(message, color=Fore.CYAN):
    if DEBUG_MODE:
        print(color + message + Fore.RESET)

# Configuration
CONFIG = {
    "DUCKDB_FILE_PATH": 'stock_data.duckdb',
    "FLUSH_INTERVAL": 10,
    "MAX_BUFFER_SIZE": 1000
}

# In-memory buffer for storing data temporarily
data_buffer = []
buffer_lock = Lock()

def setup_duckdb():
    """Setup the DuckDB database and create the table if it doesn't exist."""
    with duckdb.connect(CONFIG["DUCKDB_FILE_PATH"]) as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                symbol TEXT,
                time INTEGER,
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
                last_transaction_price REAL,
                last_transaction_volume INTEGER,
                last_transaction_id INTEGER
            )
        ''')
        debug_log("DuckDB setup complete.", Fore.GREEN)

setup_duckdb()

def generate_websocket_key():
    """Generate a base64-encoded WebSocket key."""
    return base64.b64encode(os.urandom(16)).decode('utf-8')

async def store_data(data):
    """Store the received data into the in-memory buffer."""
    global data_buffer
    try:
        if not data or not isinstance(data, dict):
            return

        last_transaction = data.get("lt")
        last_transaction_id = last_transaction.get('t') if last_transaction else None
        last_transaction_price = last_transaction.get('x') if last_transaction else None
        last_transaction_volume = last_transaction.get('v') if last_transaction else None

        # Ensure required keys are present
        required_keys = ['s', 't', 'o', 'h', 'l', 'c', 'v']
        if not all(key in data for key in required_keys):
            debug_log(f"Missing required fields in data: {data}", Fore.RED)
            return

        # Prepare the record to match the DuckDB schema
        record = (
            data['s'],                    # symbol
            data['t'],                    # time
            data['o'],                    # open
            data['h'],                    # high
            data['l'],                    # low
            data['c'],                    # close
            data['v'],                    # volume
            data.get('ldcp'),             # last close
            data.get('ch'),               # change
            round(data.get('pch', 0) * 100, 2),  # change percent
            data.get('bp'),               # bid price
            data.get('bv'),               # bid volume
            data.get('ap'),               # ask price
            data.get('av'),               # ask volume
            data.get('val'),              # total value
            data.get('tr'),               # transactions
            last_transaction_price,       # last transaction price
            last_transaction_volume,      # last transaction volume
            last_transaction_id           # last transaction ID
        )

        async with buffer_lock:
            data_buffer.append(record)
            debug_log(f"Data stored in buffer. Buffer size: {len(data_buffer)}", Fore.GREEN)

    except Exception as e:
        debug_log(f"Error storing data: {e}. Data: {data}", Fore.RED)

async def flush_data_to_duckdb():
    """Flush the buffered data to the DuckDB database."""
    global data_buffer
    async with buffer_lock:
        if not data_buffer:
            return

        try:
            debug_log(f"Flushing {len(data_buffer)} records to DuckDB...", Fore.CYAN)

            # Convert the buffer to a DataFrame
            df = pd.DataFrame(data_buffer, columns=[
                "symbol", "time", "open", "high", "low", "close", "volume",
                "last_close", "change", "change_percent", "bid_price", "bid_volume",
                "ask_price", "ask_volume", "total_value", "transactions",
                "last_transaction_price", "last_transaction_volume", "last_transaction_id"
            ])

            with duckdb.connect(CONFIG["DUCKDB_FILE_PATH"]) as db:
                db.execute("INSERT INTO stock_data SELECT * FROM df")

            data_buffer.clear()
            debug_log("Data successfully flushed to DuckDB.", Fore.GREEN)

        except Exception as e:
            debug_log(f"Error during flush: {e}", Fore.RED)

async def periodic_flush():
    """Periodically flush data from the buffer to the DuckDB database."""
    while True:
        await asyncio.sleep(CONFIG["FLUSH_INTERVAL"])
        await flush_data_to_duckdb()

async def connect():
    """Connect to WebSocket server and handle incoming messages."""
    uri = "wss://market.capitalstake.com/stream"
    headers = {"sec-websocket-key": generate_websocket_key()}

    while True:
        try:
            async with websockets.connect(
                uri, extra_headers=headers, ping_interval=20, ping_timeout=10
            ) as websocket:
                debug_log("Connected to WebSocket!", Fore.GREEN)

                asyncio.create_task(periodic_flush())

                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    debug_log(f"Received data: {data}")
                    if data.get("t") == "tick" and "d" in data:
                        await store_data(data["d"])

        except websockets.ConnectionClosedError as e:
            debug_log(f"WebSocket connection closed: {e}. Reconnecting...", Fore.RED)
        except Exception as e:
            debug_log(f"Unexpected connection error: {e}. Reconnecting in 5 seconds...", Fore.RED)
            await asyncio.sleep(5)

def main():
    try:
        asyncio.run(connect())
    except KeyboardInterrupt:
        debug_log("Stream interrupted by user. Exiting...", Fore.YELLOW)
    finally:
        asyncio.run(flush_data_to_duckdb())
        debug_log("Final data flush completed.", Fore.GREEN)

if __name__ == "__main__":
    main()
