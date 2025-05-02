#!/usr/bin/env python3
import socket
import argparse
import datetime

def run_server(port: int):
    """Listen for one connection, receive client's time, then send server time back."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(('', port))
        srv.listen(1)
        print(f"[SERVER] Listening on port {port}…")
        conn, addr = srv.accept()
        with conn:
            print(f"[SERVER] Connected by {addr}")
            # 1) receive client time
            data = conn.recv(1024)
            client_time = data.decode('utf-8')
            print(f"[SERVER] Client time received: {client_time}")
            # 2) send server time
            server_time = datetime.datetime.now().isoformat()
            conn.sendall(server_time.encode('utf-8'))
            print(f"[SERVER] Server time sent:     {server_time}")

def run_client(host: str, port: int):
    """Connect to server, send client time, then receive server time back."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cli:
        cli.connect((host, port))
        # 1) send client time
        client_time = datetime.datetime.now().isoformat()
        cli.sendall(client_time.encode('utf-8'))
        print(f"[CLIENT] Client time sent:     {client_time}")
        # 2) receive server time
        data = cli.recv(1024)
        server_time = data.decode('utf-8')
        print(f"[CLIENT] Server time received: {server_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exchange current time between two devices over TCP."
    )
    parser.add_argument(
        "-m", "--mode", choices=["server", "client"], required=True,
        help="Run as server or client."
    )
    parser.add_argument(
        "-H", "--host", default="127.0.0.1",
        help="Server IP address (client mode only)."
    )
    parser.add_argument(
        "-p", "--port", type=int, default=12345,
        help="TCP port to use (default: 12345)."
    )
    args = parser.parse_args()

    if args.mode == "server":
        run_server(args.port)
    else:
        run_client(args.host, args.port)
