#!/usr/bin/env python3
import socket
import argparse
import datetime
import time
import math
from zyx_multi_device import start_cameras

def compute_start_second(received_time_iso: str, offset: float = 3.0) -> float:
    """Parse ISO time, add offset, then round up to next integer second (0–59)."""
    dt = datetime.datetime.fromisoformat(received_time_iso)
    secs = dt.second + dt.microsecond / 1e6
    raw = secs + offset
    # wrap around minute
    start_sec = math.ceil(raw) % 60
    return float(start_sec)

def schedule_print(role: str, target_sec: float):
    """
    Sleep until the next occurrence of target_sec within the minute,
    then print local datetime.
    """
    now = datetime.datetime.now()
    cur = now.second + now.microsecond / 1e6
    if target_sec >= now.second:
        wait = (target_sec - now.second) - now.microsecond / 1e6
    else:
        # next minute
        wait = (60 - cur) + target_sec
    if wait > 0:
        print(f"[{role}] sleeping {wait:.3f}s until second {target_sec:.6f}")
        time.sleep(wait)
    print(f"[{role} START] {datetime.datetime.now().isoformat()}")
    start_cameras()

def run_server(port: int) -> float:
    """Wait for client, receive its time, compute/send start_sec, then return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(('', port))
        srv.listen(1)
        print(f"[SERVER] Listening on port {port} …")
        conn, addr = srv.accept()
        with conn:
            print(f"[SERVER] Connected by {addr}")
            data = conn.recv(1024)
            client_time = data.decode('utf-8')
            print(f"[SERVER] Received client time: {client_time}")
            start_sec = compute_start_second(client_time)
            conn.sendall(f"{start_sec:.6f}".encode('utf-8'))
            print(f"[SERVER] Sent start second: {start_sec:.6f}")
    return start_sec

def run_client(host: str, port: int) -> float:
    """Connect to server, send our time, receive start_sec, then return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cli:
        cli.connect((host, port))
        now_iso = datetime.datetime.now().isoformat()
        cli.sendall(now_iso.encode('utf-8'))
        print(f"[CLIENT] Sent client time: {now_iso}")
        data = cli.recv(1024)
        start_sec = float(data.decode('utf-8'))
        print(f"[CLIENT] Received start second: {start_sec:.6f}")
    return start_sec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronize a start time (seconds-only) between two devices."
    )
    parser.add_argument("-m", "--mode", choices=["server", "client"], required=True, help="server waits; client initiates")
    parser.add_argument("-H", "--host", default="127.0.0.1", help="server IP (client mode only)")
    parser.add_argument("-p", "--port", type=int, default=12345, help="TCP port (default: 12345)")
    args = parser.parse_args()

    if args.mode == "server":
        start_sec = run_server(args.port)
        schedule_print("SERVER", start_sec)
    else:
        start_sec = run_client(args.host, args.port)
        schedule_print("CLIENT", start_sec)
