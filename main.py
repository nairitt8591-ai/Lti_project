import subprocess

def run_script(script_name):
    print(f"\n🚀 Running {script_name}...")
    result = subprocess.run(["python", script_name])
    if result.returncode == 0:
        print(f"✅ {script_name} completed successfully.")
        return True
    else:
        print(f"❌ {script_name} failed with exit code {result.returncode}.")
        return False

if __name__ == "__main__":
    # Step 1: Data Preprocessing
    if run_script("data_preprocessing.py"):
        # Step 2: Model Training
        if run_script("train.py"):
            print("\n🎉 Pipeline completed successfully!")
        else:
            print("\n⚠️ Training failed. Check train.py for errors.")
    else:
        print("\n⚠️ Preprocessing failed. Training skipped.")