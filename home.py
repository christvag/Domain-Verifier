from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import threading
import time
import subprocess
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
app.config['PROCESSED_FOLDER'] = 'all/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

progress = {'percent': 0, 'log': []}
processed_filename = None
stop_event = threading.Event()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def run_verification(filepath, retries):
    global progress, processed_filename, stop_event
    progress['percent'] = 0
    progress['log'] = []
    processed_filename = None
    stop_event.clear()
    # Build the command to run verify_websites.py
    cmd = [sys.executable, 'verify_websites.py', filepath, '--retries', str(retries)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    total_percent = 0
    for line in process.stdout:
        if stop_event.is_set():
            process.terminate()
            progress['log'].append('Process stopped by user.')
            break
        line = line.strip()
        # Try to parse progress from script output
        if 'Verifying' in line and '%' in line:
            # Example: Verifying (HEAD)  50%|#####     | 5/10
            import re
            m = re.search(r'(\d+)%', line)
            if m:
                total_percent = int(m.group(1))
                progress['percent'] = total_percent
        progress['log'].append(line)
        if len(progress['log']) > 100:
            progress['log'] = progress['log'][-100:]
    process.wait()
    # After process, try to find the output file
    import glob
    base = os.path.splitext(os.path.basename(filepath))[0]
    report_files = glob.glob(os.path.join(app.config['PROCESSED_FOLDER'], f'{base}_report_all.csv'))
    if report_files:
        processed_filename = report_files[0]
    progress['percent'] = 100
    progress['log'].append('Verification complete.')
    # Clear the upload folder after verification is complete
    upload_folder = app.config['UPLOAD_FOLDER']
    for f in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, f)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception:
            pass


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    retries = int(request.form.get('retries', 1))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Start verification in a thread
        thread = threading.Thread(target=run_verification, args=(filepath, retries))
        thread.start()
        return '', 204
    return redirect(url_for('index'))


@app.route('/progress')
def get_progress():
    return jsonify(progress)


@app.route('/download')
def download_file():
    import os  # Ensure os is imported for this function
    global processed_filename
    req_file = request.args.get('file')
    folder = request.args.get('folder', 'all')
    # Determine the folder path
    if folder not in ['all', 'reachable', 'upload']:
        return '', 404
    folder_path = os.path.join(os.getcwd(), folder)
    if req_file:
        file_path = os.path.join(folder_path, req_file)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        return '', 404
    # If no file specified, try to send the latest file in the folder
    import glob
    files = glob.glob(os.path.join(folder_path, '*.csv'))
    if files:
        latest_file = max(files, key=os.path.getmtime)
        return send_file(latest_file, as_attachment=True)
    return '', 404


@app.route('/stop', methods=['POST'])
def stop_process():
    global stop_event
    stop_event.set()
    return '', 204


@app.route('/clear_log', methods=['POST'])
def clear_log():
    log_path = os.path.join(os.getcwd(), 'process.log')
    try:
        with open(log_path, 'w') as f:
            f.write('')
    except Exception:
        pass
    progress['log'] = []
    return '', 204


def get_processed_files(folder=None):
    import glob
    import pandas as pd
    from datetime import datetime
    import os
    files = []
    if folder is None:
        folder = app.config['PROCESSED_FOLDER']
    else:
        # If folder is an absolute path, use as is; otherwise, join with app root
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)
    print(f"[DEBUG] Looking for files in folder: {folder}")
    if not os.path.exists(folder):
        print(f"[DEBUG] Folder does not exist: {folder}")
        return []
    csv_paths = glob.glob(os.path.join(folder, '*.csv'))
    print(f"[DEBUG] Found CSV files: {csv_paths}")
    for path in csv_paths:
        fname = os.path.basename(path)
        # Get file date
        mtime = os.path.getmtime(path)
        date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        eta = ''
        total = ''
        good = ''
        try:
            df = pd.read_csv(path)
            # For report_all.csv, total = unique websites checked, good = reachable==True
            if fname.endswith('_report_all.csv'):
                total = int(len(df))
                if 'reachable' in df.columns:
                    col = df['reachable']
                    if col.dtype == bool:
                        good = int(col.sum())
                    elif col.dtype == int or col.dtype == float:
                        good = int((col == 1).sum())
                    else:
                        good = int(col.astype(str).str.lower().isin(['true', '1']).sum())
                else:
                    good = ''
            else:
                # For other files, fallback to previous logic
                total = int(len(df))
                if 'reachable' in df.columns:
                    col = df['reachable']
                    if col.dtype == bool:
                        good = int(col.sum())
                    elif col.dtype == int or col.dtype == float:
                        good = int((col == 1).sum())
                    else:
                        good = int(col.astype(str).str.lower().isin(['true', '1']).sum())
                else:
                    good = ''
            eta = 'N/A'
        except Exception:
            pass
        files.append({
            'filename': fname,
            'date': date_str,
            'eta': eta,
            'total': total,
            'good': good
        })
    files.sort(key=lambda x: x['date'], reverse=True)
    return files


@app.route('/log')
def log_table():
    folder = request.args.get('folder')
    files = get_processed_files(folder)
    return jsonify(files)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    app.run(debug=True)
