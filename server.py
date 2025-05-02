import socket
from audio_denoise import denoise_audio

# Server settings
HOST = '127.0.0.1'
PORT = 65432

def start_server():
    print("Starting server...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        s.settimeout(1.0)  # Set timeout of 1 second for accept()
        print(f"Server listening on {HOST}:{PORT}")
        print("\nPress Ctrl+C to terminate the server...")

        try:
            while True:
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue  # Timeout expired, check for KeyboardInterrupt
                with conn:
                    print(f"Connected by {addr}")
                    data = conn.recv(1024)
                    if not data:
                        continue
                    command = data.decode()

                    if command == 'denoise':
                        print("Received denoise command. Starting denoising process...")
                        denoise_audio()
                        print("Denoising completed. Sending back success signal...")
                        conn.sendall(b'done')
                    else:
                        print(f"Unknown command: {command}")
                        conn.sendall(b'error')
        except KeyboardInterrupt:
            print("\nServer terminated by user ")

if __name__ == "__main__":
    start_server()