import subprocess
import os
import sys
import time

def kill_process_on_port(port):
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                subprocess.run(["kill", "-9", pid])
                print(f"✔️ Matou processo na porta {port} (PID {pid})")
    except Exception as e:
        print(f"Erro ao tentar matar processo na porta {port}: {e}")


def run_fastapi():
    subprocess.Popen([sys.executable, 'routes/routes.py']) 

def run_streamlit():
    subprocess.Popen(['streamlit', 'run', 'streamlitPages/page1.py'])

def run_mlflow():
    subprocess.Popen(['mlflow', 'ui']) 

if __name__ == '__main__':
    ports = [8501, 5000, 8000]
    for port in ports:
        kill_process_on_port(port)

    run_fastapi()
    run_streamlit()
    run_mlflow()

    print("Todos os serviços estão rodando...")
    print("Fast API em: http://127.0.0.1:8000/")
    print("Streamlit em: http://127.0.0.1:8501/")
    print("Mlflow em: http://127.0.0.1:5000/")

    try:
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nEncerrando container...")