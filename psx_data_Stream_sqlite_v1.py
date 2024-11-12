import asyncio
import websockets
import sqlite3
import json
import base64
import os
from datetime import datetime
from colorama import init, Fore

# Initialize Colorama for colored terminal output
init(autoreset=True)

# Global control for enabling/disabling debug logs
DEBUG_MODE = False  # Set to False to disable debug logs

# Function to print debug logs based on the DEBUG_MODE
def debug_log(message, color=Fore.CYAN):
    if DEBUG_MODE:
        print(color + message + Fore.RESET)

# Connect to the SQLite database
conn = sqlite3.connect('stock_data.db', check_same_thread=False)
cursor = conn.cursor()

# Create the stock_data table if it doesn't exist
cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS stock_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
conn.commit()

# In-memory buffer for storing data temporarily
data_buffer = []
buffer_lock = asyncio.Lock()
FLUSH_INTERVAL = 10  # Time interval in seconds to flush data to the database

def generate_websocket_key():
    random_bytes = os.urandom(16)
    return base64.b64encode(random_bytes).decode('utf-8')

async def flush_data_to_db():
    """Flush the buffered data to the database."""
    global data_buffer
    async with buffer_lock:
        if not data_buffer:
            debug_log("No data to flush.", Fore.YELLOW)
            return

        try:
            debug_log(f"Flushing {len(data_buffer)} records to the database...")
            cursor.executemany(''' 
                INSERT INTO stock_data (
                    symbol, time, open, high, low, close, volume, last_close, change,
                    change_percent, bid_price, bid_volume, ask_price, ask_volume,
                    total_value, transactions, last_transaction_price, last_transaction_volume,
                    last_transaction_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_buffer)
            conn.commit()
            data_buffer.clear()
            debug_log("Data successfully flushed to the database.", Fore.GREEN)
        except sqlite3.Error as e:
            debug_log(f"Database error during flush: {e}", Fore.RED)
            conn.rollback()
        except Exception as e:
            debug_log(f"Unexpected error during flush: {e}", Fore.RED)

async def store_data(data):
    """Store the received data into the in-memory buffer."""
    global data_buffer
    try:
        if not data or not isinstance(data, dict):
            debug_log(f"Invalid data received: {data}", Fore.YELLOW)
            return

        last_transaction = data.get("lt")
        # Extract values correctly from the 'lt' dictionary
        last_transaction_id = last_transaction.get('t') if last_transaction else None
        last_transaction_price = last_transaction.get('x') if last_transaction else None
        last_transaction_volume = last_transaction.get('v') if last_transaction else None

        # Ensure required keys are present before creating a record
        required_keys = ['s', 't', 'o', 'h', 'l', 'c', 'v']
        if not all(key in data for key in required_keys):
            debug_log(f"Missing required fields in data: {data}", Fore.RED)
            return

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
            last_transaction_id           # last transaction ID (timestamp)
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

async def periodic_flush():
    """Periodically flush data from the buffer to the database."""
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL)
            await flush_data_to_db()
        except Exception as e:
            debug_log(f"Error in periodic flush: {e}", Fore.RED)

async def connect():
    """Connect to the WebSocket server and handle incoming messages."""
    uri = "wss://market.capitalstake.com/stream"
    headers = {
        "sec-websocket-key": generate_websocket_key(),
        "sec-websocket-version": "13"
    }

    while True:
        try:
            async with websockets.connect(uri, extra_headers=headers, ping_interval=20, ping_timeout=10) as websocket:
                debug_log("Connected to WebSocket!", Fore.GREEN)

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

# Run the WebSocket connection
try:
    asyncio.run(connect())
except KeyboardInterrupt:
    debug_log("Stream interrupted by user. Exiting...", Fore.YELLOW)
finally:
    if conn:
        conn.close()
        debug_log("Database connection closed.", Fore.CYAN)
