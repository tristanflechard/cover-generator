from flask import Flask, render_template, request, send_file, url_for, jsonify
from core import run_infer_script
from audio_separator.separator import Separator  # Replaced import
import os
import subprocess
from youtube import youtube_to_wav
from join import mix_wav_files
import logging
from datetime import datetime

processing_status = {}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()
    ]
)

UVR_BASE_PATH = "uvr/"
APPLIO_BASE_PATH = "Applio/"

app = Flask(__name__)

def get_available_models():
    try:
        models_dir = f"logs/model"
        models = [model for model in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, model))]
        logging.info("Retrieved models: %s", models)
        return models
    except Exception as e:
        logging.error("Failed to retrieve models", exc_info=True)
        return []

@app.route('/status/<task_id>')
def get_status(task_id):
    return jsonify(processing_status.get(task_id, {
        'step': 0,
        'message': 'Task not found',
        'complete': False
    }))

@app.template_filter('datetime')
def format_datetime(value):
    dt = datetime.fromtimestamp(value)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

@app.route('/', methods=['GET', 'POST'])
def home():
    models = get_available_models()
    
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')
        selected_model = request.form.get('selected_model')
        
        if not selected_model:
            return render_template('index.html', error="Please select a model", models=models)
            
        if not youtube_url:
            return render_template('index.html', error="Please enter a YouTube URL", models=models)
            
        try:
            # Generate a unique task ID using timestamp
            task_id = str(datetime.now().timestamp()).replace('.', '')
            processing_status[task_id] = {
                'step': 1,
                'message': 'Starting download and WAV conversion...',
                'complete': False
            }

            # Step 1: Convert YouTube URL to WAV
            logging.info("Received YouTube URL for processing: %s", youtube_url)
            input_path = youtube_to_wav(youtube_url)
            logging.info("YouTube audio downloaded and converted to WAV: %s", input_path)
            
            processing_status[task_id] = {
                'step': 2,
                'message': 'Splitting audio...',
                'complete': False
            }
            logging.info("Splitting audio for: %s", input_path)
            
            # Step 2: Split the audio using audio_separator
            input_file = f"yt_audios/{input_path}"
            separator = Separator()
            separator.load_model(model_filename='UVR-MDX-NET-Inst_HQ_3.onnx')
            output_files = separator.separate(input_file)
            
            # Ensure output directory exists
            split_audios_dir = os.path.join('.','split_audios')
            os.makedirs(split_audios_dir, exist_ok=True)
            
            # Rename and move output files
            vocal_filename = f'vocal_{input_path}.wav'
            instrumental_filename = f'instrument_{input_path}.wav'
            vocal_path = os.path.join(split_audios_dir, vocal_filename)
            instrumental_path = os.path.join(split_audios_dir, instrumental_filename)
            
            for output_file in output_files:
                if 'vocals' in output_file.lower():
                    os.rename(output_file, vocal_path)
                elif 'instrumental' in output_file.lower() or 'accompaniment' in output_file.lower():
                    os.rename(output_file, instrumental_path)
            
            logging.info(f"Split audio into {vocal_path} and {instrumental_path}")

            processing_status[task_id] = {
                'step': 3,
                'message': 'Processing with selected model...',
                'complete': False
            }

            # Step 3: Run inference (remaining code unchanged)
            model_base_path = os.path.join("logs", "model")
            model_full_path = None
            
            for root, dirs, files in os.walk(model_base_path):
                if selected_model in dirs:
                    model_full_path = os.path.join(root, selected_model)
                    break
            
            if not model_full_path:
                raise Exception(f"Model {selected_model} not found")
            logging.info("Changing voice with model: %s", model_full_path)
            
            applio_output_name = f"processed_voices/{input_path}_FEAT_{selected_model}.wav"
            # Run inference directly with default parameters
            run_infer_script(
                pitch=0,
                filter_radius=3,
                index_rate=0.3,
                volume_envelope=1,
                protect=0.33,
                hop_length=128,
                f0_method="rmvpe",
                input_path=vocal_path,
                output_path=applio_output_name,
                pth_path=os.path.join(model_full_path, "model.pth"),
                index_path=os.path.join(model_full_path, "model.index"),
                split_audio=False,
                f0_autotune=False,
                f0_autotune_strength=1.0,
                clean_audio=False,
                clean_strength=0.7,
                export_format="WAV",
                f0_file=None,
                embedder_model="contentvec",
                embedder_model_custom=None,
                formant_shifting=False,
                formant_qfrency=1.0,
                formant_timbre=1.0,
                post_process=False,
                reverb=False,
                pitch_shift=False,
                limiter=False,
                gain=False,
                distortion=False,
                chorus=False,
                bitcrush=False,
                clipping=False,
                compressor=False,
                delay=False,
                reverb_room_size=0.5,
                reverb_damping=0.5,
                reverb_wet_gain=0.5,
                reverb_dry_gain=0.5,
                reverb_width=0.5,
                reverb_freeze_mode=0.5,
                pitch_shift_semitones=0.0,
                limiter_threshold=-6,
                limiter_release_time=0.01,
                gain_db=0.0,
                distortion_gain=25,
                chorus_rate=1.0,
                chorus_depth=0.25,
                chorus_center_delay=7,
                chorus_feedback=0.0,
                chorus_mix=0.5,
                bitcrush_bit_depth=8,
                clipping_threshold=-6,
                compressor_threshold=0,
                compressor_ratio=1,
                compressor_attack=1.0,
                compressor_release=100,
                delay_seconds=0.5,
                delay_feedback=0.0,
                delay_mix=0.5,
                sid=0
            )

            logging.info("Inference completed for: %s", input_path)

            logging.info("Mixing final audio")

            processing_status[task_id] = {
                'step': 4,
                'message': 'Mixing final audio...',
                'complete': False
            }

            # Step 4: Mix the WAV files (remaining code unchanged)
            mix_wav_files(
                f"split_audios/instrument_{input_path}.wav",
                f"{applio_output_name}",
                f"outputs/{input_path}_FEATURING_{selected_model}.wav"
            )
            logging.info("Mixing completed: %s", input_path)

            processing_status[task_id] = {
                'step': 5,
                'message': 'Processing complete!',
                'complete': True
            }

            return render_template('index.html', 
                                task_id=task_id,
                                message=f"Processing started! Task ID: {task_id}", 
                                models=models)
        except Exception as e:
            logging.error("An error occurred during processing", exc_info=True)
            if task_id in processing_status:
                processing_status[task_id] = {
                    'step': 0,
                    'message': f'Error: {str(e)}',
                    'complete': True
                }
            return render_template('index.html', error=str(e), models=models)

    return render_template('index.html', models=models)

# Remaining routes and code unchanged...

@app.route('/covers')
def list_covers():
    try:
        output_dir = f"outputs"
        covers = []
        
        # Get all WAV files from the outputs directory
        for filename in os.listdir(output_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(output_dir, filename)
                file_stats = os.stat(file_path)
                cover_info = {
                    'filename': filename,
                    'path': file_path,
                    'size': round(file_stats.st_size / (1024 * 1024), 2),  # Size in MB
                    'created': os.path.getctime(file_path),
                }
                covers.append(cover_info)
        
        # Sort covers by creation date, newest first
        covers.sort(key=lambda x: x['created'], reverse=True)
        
        return render_template('covers.html', covers=covers)
    except Exception as e:
        logging.error("Error loading covers", exc_info=True)
        return render_template('covers.html', error=str(e))

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join('outputs/', filename)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logging.error(f"Error downloading file {filename}", exc_info=True)
        return str(e), 404

# Add this route to serve audio files for the audio player
@app.route('/play/<filename>')
def serve_audio(filename):
    try:
        file_path = os.path.join('outputs/', filename)
        return send_file(file_path, mimetype='audio/wav')
    except Exception as e:
        logging.error(f"Error serving audio file {filename}", exc_info=True)
        return str(e), 404


if __name__ == '__main__':
    app.run(debug=True)