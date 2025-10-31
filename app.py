import os
import json
import time
import random
import shutil
import base64
import tempfile
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import logging


import subprocess



# from src.Services.rubberband_selective_processor_v2 import normalize_audio_selectively_v2 as process_r2
# from src.Services.R3_selective_processor import normalize_audio_selectively_v2 as process_r3


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)


STORAGE_PATH = os.getenv('STORAGE_PATH', 'storage')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
VOICE_ID = os.getenv('VOICE_ID', 'GUDYcgRAONiI1nXDcNQQ')
MODEL_ID = os.getenv('MODEL_ID', 'eleven_multilingual_v2')
ENABLE_PYTHON_PROCESSING = os.getenv('ENABLE_PYTHON_PROCESSING', 'true').lower() == 'true'
PYTHON_SERVICE_URL = os.getenv('PYTHON_SERVICE_URL', 'http://python-audio-processor:10000')

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY is required")
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is required")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY is required")


def generate_project_id() -> str:
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    random_suffix = ''.join(random.choices('0123456789abcdef', k=6))
    return f"{timestamp}-{random_suffix}"

def delete_folder(folder_path: str) -> bool:
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting folder {folder_path}: {e}")
        return False

def ensure_directory(path: str) -> bool:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False

class ExcelService:
    def read(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            df = pd.read_excel(file_path)
            columns = df.columns.tolist()
            rows = df.to_dict('records')
            return {
                'columns': columns,
                'rows': rows
            }
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return None
    
    def prepare_text(self, rows: List[Dict], columns: List[str], column_name: str, start_row: int = 4) -> Optional[str]:
        if column_name not in columns:
            return None
        entries = []
        for i, row in enumerate(rows):
            row_number = i + 2 
            if row_number < start_row:
                continue
            if column_name not in row:
                continue
            value = row[column_name]
            if value is None or pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                entries.append(text)
        if not entries:
            return None
        combined = ". ".join(entries)
        combined = combined.replace(". ", "... ")
        return combined

class TTSService:
    def __init__(self):
        self.api_key = ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io"
        self.session = requests.Session()
        self.session.headers.update({
            'xi-api-key': self.api_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def get_remaining_credits(self) -> Optional[int]:
        try:
            response = self.session.get(f"{self.base_url}/v1/user")
            response.raise_for_status()
            data = response.json()
            used = data.get('subscription', {}).get('character_count', 0)
            limit = data.get('subscription', {}).get('character_limit', 0)
            if limit <= 0:
                logger.warning("No character limit found in response")
                return None
            
            return max(0, limit - used)
        except Exception as e:
            logger.error(f"Error getting remaining credits: {e}")
            return None
    
    def synthesize_with_alignment(self, voice_id: str, model_id: str, text: str, voice_settings) -> Optional[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/v1/text-to-speech/{voice_id}/with-timestamps"
            
            payload = {
                "model_id": model_id,
                "text": text,
                "output_format": "mp3_44100_192",
                "voice_settings": voice_settings,
                "return_alignment": True
            }
            
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

class AudioService:
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.dirname(__file__))

    def get_duration(self, file_path: str) -> float:
        try:
            import subprocess
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return float(result.stdout.strip())
            return 0.0
        except Exception as e:
            logger.error(f"Error getting duration for {file_path}: {e}")
            return 0.0
    
    def run_ffmpeg_command(self, cmd: List[str]) -> bool:
        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                logger.error(f"FFmpeg command failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error running ffmpeg command: {e}")
            return False
    
    def stitch(self, speech_dir: str, output_dir: str, opts: Dict[str, Any]) -> Optional[Dict[str, str]]:
        try:
            # Define sequence
            sequence = [
                '1',
                ['2a', '2b'],
                '3',
                ['4a', '4b'],
                '5',
                ['6a', '6b'],
                '7',
                ['8a', '8b'],
                '9',
                ['10a', '10b'],
                '11',
                ['12a', '12b'],
                '13',
                '14'
            ]
            
            intro_outro_dir = os.path.join(self.root_dir, 'intro-outro-audio')
            effects_dir = os.path.join(self.root_dir, 'effects-audio')
            root_path = os.path.abspath(os.path.dirname(__file__))
            
            temp_dir = os.path.join(root_path, STORAGE_PATH, 'projects', output_dir, 'tmp')
            ensure_directory(temp_dir)
            
            bell = os.path.join(effects_dir, 'bell-15s.mp3')
            intro = os.path.join(intro_outro_dir, 'column_1.mp3')
            outro = os.path.join(intro_outro_dir, 'column_14.mp3')
            
            voice_speed = float(opts.get('voice_speed', 0.9))
            bell_volume = float(opts.get('bell_volume', 0.6))
            pause_before = float(opts.get('pause_before_bell', 2.0))
            pause_after = float(opts.get('pause_after_bell', 2.0))
            
            inputs = []
            filters = []
            concat_inputs = []
            input_index = 0
            
            for idx, item in enumerate(sequence):
                if isinstance(item, list):
                    a, b = item
                    file_a = os.path.join(root_path, STORAGE_PATH, 'projects', speech_dir, f'column_{a}_normalized.wav')
                    file_b = os.path.join(root_path, STORAGE_PATH, 'projects', speech_dir, f'column_{b}_normalized.wav')
                    
                    if not os.path.isfile(file_a) or not os.path.isfile(file_b):
                        continue
                    
                    dur_a = self.get_duration(file_a)
                    dur_b = self.get_duration(file_b)
                    pad_dur = max(0, abs(dur_a - dur_b))
                    
                    inputs.extend(['-i', file_a, '-i', file_b])
                    
                    filters.append(f"[{input_index}]asetrate=44100*{voice_speed},aresample=44100,atempo=1,apad=pad_dur={pad_dur}[a{idx}]")
                    filters.append(f"[{input_index + 1}]asetrate=44100*{voice_speed},aresample=44100,atempo=1,apad=pad_dur={pad_dur}[b{idx}]")

                    # filters.append(f"[a{idx}][b{idx}]amerge=inputs=2[track{idx}]")
                    filters.append(f"[a{idx}][b{idx}]concat=n=2:v=0:a=1[track{idx}]")

                    concat_inputs.append(f"[track{idx}]")
                    input_index += 2
                else:
                    col = str(item)
                    src = os.path.join(root_path, STORAGE_PATH, 'projects', speech_dir, f'column_{col}_normalized.wav')
                    
                    if not os.path.isfile(src):
                        continue
                    
                    inputs.extend(['-i', src])
                    filters.append(f"[{input_index}]asetrate=44100*{voice_speed},aresample=44100,atempo=1,"
                                 f"pan=stereo|c0=c0|c1=c0[track{idx}]")
                    
                    section_num = idx + 1
                    if os.path.isfile(bell) and (section_num % 2 == 1):
                        inputs.extend(['-i', bell])
                        filters.append(f"[{input_index + 1}]volume={bell_volume}[bell{idx}]")
                        filters.append(f"anullsrc=r=44100:cl=stereo:d={pause_before}[silb_before{idx}]")
                        filters.append(f"anullsrc=r=44100:cl=stereo:d={pause_after}[silb_after{idx}]")
                        
                        concat_inputs.extend([f"[silb_before{idx}]", f"[bell{idx}]", f"[silb_after{idx}]", f"[track{idx}]"])
                        input_index += 2
                    else:
                        concat_inputs.append(f"[track{idx}]")
                        input_index += 1
            
            filter_str = ";".join(filters) + ";" + "".join(concat_inputs) + f"concat=n={len(concat_inputs)}:v=0:a=1[out]"
            
            voice_track = os.path.join(root_path, STORAGE_PATH, 'projects', output_dir, 'hypnosis_voice.mp3')
            
            cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', filter_str, '-map', '[out]', 
                   '-c:a', 'libmp3lame', '-b:a', '192k', voice_track]
            
            if not self.run_ffmpeg_command(cmd):
                return None
            
            voice_norm = os.path.join(temp_dir, 'voice_norm.mp3')
            norm_cmd = ['ffmpeg', '-y', '-i', voice_track, '-filter:a', 'loudnorm', voice_norm]
            
            if self.run_ffmpeg_command(norm_cmd):
                shutil.move(voice_norm, voice_track)
            
            return {'voice': voice_track}
            
        except Exception as e:
            logger.error(f"Audio stitching failed: {e}")
            return None

class PythonProcessorService:
    
    def __init__(self):
        self.service_url = PYTHON_SERVICE_URL
    
    def process_audio(self, audio_file_path: str, alignment_data: Dict[str, Any], engine: str = 'r2') -> Optional[Dict[str, Any]]:
        if not os.path.isfile(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return None
        
        try:
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            files = {
                'audio_file': ('audio.mp3', audio_data, 'audio/mpeg')
            }
            
            data = {
                'alignment_data': json.dumps(alignment_data),
                'engine': engine,
                'output_format': 'mp3'
            }
            
            response = requests.post(
                f"{self.service_url}/api/process",
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                return {
                    'success': True,
                    'processed_audio': response.content,
                    'content_type': response.headers.get('content-type')
                }
            else:
                logger.error(f"Python service error: HTTP {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Python service processing failed: {e}")
            return None
    
    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.service_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'ok'
            return False
        except Exception as e:
            logger.error(f"Python service health check failed: {e}")
            return False

class SupabaseService:
    
    def __init__(self):
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        self.bucket = "audio-sessions"
    
    def upload_file(self, file_data: bytes, file_name: str) -> Dict[str, Any]:
        try:
            upload_url = f"{self.url}/storage/v1/object/{self.bucket}/{file_name}"
            
            headers = {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "audio/mpeg"
            }
            
            response = requests.post(upload_url, data=file_data, headers=headers, timeout=300)
            
            return {
                'success': response.status_code >= 200 and response.status_code < 300,
                'http_code': response.status_code,
                'response': response.text
            }
        except Exception as e:
            logger.error(f"Supabase upload failed: {e}")
            return {
                'success': False,
                'http_code': 500,
                'response': str(e)
            }
    
    def insert_audio_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            url = f"{self.url}/rest/v1/hy_audio"
            
            headers = {
                "Authorization": f"Bearer {self.key}",
                "apikey": self.key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=300)
            
            return {
                'success': response.status_code >= 200 and response.status_code < 300,
                'http_code': response.status_code,
                'response': response.json() if response.content else {}
            }
        except Exception as e:
            logger.error(f"Database insert failed: {e}")
            return {
                'success': False,
                'http_code': 500,
                'response': {'error': str(e)}
            }


def process_r2(out_file, processed_path, alignment_file, target_cps=12.0):
    try:
        with open(alignment_file, "r", encoding="utf-8") as f:
            alignment = json.load(f)

        chars = alignment["alignment"]["characters"]
        starts = alignment["alignment"]["character_start_times"]
        ends = alignment["alignment"]["character_end_times"]

        if not chars or not starts or not ends:
            print("Alignment data invalid — skipping.")
            return

        duration = ends[-1] - starts[0]
        total_chars = len(chars)
        actual_cps = total_chars / duration

        print(f"Normalizing speed:")
        print(f"   Duration: {duration:.1f}s | Chars: {total_chars} | CPS: {actual_cps:.2f}")

        tempo_ratio = actual_cps / target_cps

        tempo_ratio = max(0.7, min(tempo_ratio, 1.3))
        print(f"   → Target CPS: {target_cps:.2f} | Tempo ratio: {tempo_ratio:.2f}")

        processed_path = Path(processed_path)
        processed_path.mkdir(parents=True, exist_ok=True)

        output_wav = processed_path / f"{Path(out_file).stem}_normalized.wav"

        cmd = [
            "rubberband",
            "--fine",            
            "--formant",       
            "--crisp", "5",     
            "-T", f"{tempo_ratio:.3f}",
            str(out_file),
            str(output_wav)
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"Saved normalized file: {output_wav}")
        return str(output_wav)

    except subprocess.CalledProcessError as e:
        print(f"Rubberband error: {e.stderr.decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"Process error: {e}")


def process_audio_request() -> Dict[str, Any]:
    try:
        # kiễm tra có file không
        if 'file' not in request.files:
            return {'error': 'Missing file (field name: file)'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        # tạo foder storage và kiễm tra file excel
        project_id = generate_project_id()
        root_path = os.path.abspath(os.path.dirname(__file__))
        project_path = os.path.join(root_path, STORAGE_PATH, 'projects', project_id)
        speech_path = os.path.join(project_path, 'speech')
        output_path = os.path.join(project_path, 'final-output')
        
        for directory in [speech_path, output_path]:
            if not ensure_directory(directory):
                return {'error': f'Failed to create directory: {directory}'}, 500
        excel_path = os.path.join(project_path, 'source.xlsx')
        file.save(excel_path)
        excel_service = ExcelService()
        df = excel_service.read(excel_path)
        if df is None:
            return {'error': 'Failed to read Excel'}, 400
        
        # filter columns
        columns = df['columns']
        rows = df['rows']
        excluded = [] 
        requested_columns = request.form.get('columns', '').strip()
        if requested_columns:
            requested = [col.strip() for col in requested_columns.split(',') if col.strip()]
            available = [col for col in requested if col in columns and col not in excluded]
        else:
            available = [col for col in columns if col not in excluded]
        
        #khởi tạo ttsserver và xử lý text columns
        tts_service = TTSService()
        remaining = tts_service.get_remaining_credits()
        estimate = 0
        to_process = []
        for col_name in available:
            out_file = os.path.join(speech_path, f'column_{col_name}.mp3')
            if os.path.isfile(out_file):
                continue
            
            text = excel_service.prepare_text(rows, columns, col_name, 5)
            if not text or not text.strip():
                continue
            
            estimate += len(text)
            to_process.append([col_name, text, out_file])
        
        # check elevenlab còn đủ quota để sữ lý không
        if remaining is not None and estimate > remaining:
            return {
                'error': 'quota_exceeded_predicted',
                'message': 'Estimated characters exceed remaining credits',
                'remaining': remaining,
                'estimated': estimate,
                'hint': 'Reduce selected columns or upgrade quota'
            }, 402

        #phần xử lý gọi elevenlab và chuẩn hóa tốc độ
        batch_size = 2
        processed_path = os.path.join(speech_path, "processed")
        os.makedirs(processed_path, exist_ok=True)
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]
            for col_name, text, out_file in batch:
                result = tts_service.synthesize_with_alignment(
                    VOICE_ID,
                    MODEL_ID,
                    text,
                    voice_settings={
                        "stability": 0.8,
                        "similarity_boost": 0.7,
                        "style": 0.2,
                        "use_speaker_boost": True
                    }
                )

                if not result or 'audio_base64' not in result or 'alignment' not in result:
                    continue
                with open(out_file, 'wb') as f:
                    f.write(base64.b64decode(result['audio_base64']))
                if result["alignment"] and result["alignment"]["characters"]:
                    duration = result["alignment"]["character_end_times_seconds"][-1]
                    total_chars = len(result["alignment"]["characters"])
                    avg_cps = total_chars / duration
                    print(f"🎧 Duration: {duration:.1f}s | Avg CPS: {avg_cps:.1f}")
                temp_dir = tempfile.mkdtemp()
                temp_path = Path(temp_dir)
                alignment_file = temp_path / "alignment.json"

                alignment_data = { 
                    "text": text,
                    "voice_settings": {
                        "stability": 0.8,
                        "similarity_boost": 0.7,
                        "style": 0.2,
                        "use_speaker_boost": True
                    },
                    "alignment": {
                        "characters": result["alignment"]["characters"],
                        "character_start_times": result["alignment"]["character_start_times_seconds"],
                        "character_end_times": result["alignment"]["character_end_times_seconds"]
                    } if result["alignment"] else None
                }

                with open(alignment_file, 'w', encoding='utf-8') as f:
                    json.dump(alignment_data, f, indent=2, ensure_ascii=False)

                process_r2(
                    out_file,
                    processed_path,
                    alignment_file,
                    target_cps=5 # chỉnh tốc độ đọc
                )
                print(f"Done {col_name}")

        # sữ lý ghép các audio nhỏ lại thành output 
        audio_service = AudioService()
        result = audio_service.stitch(
            f"{project_id}/speech/processed",
            f"{project_id}/final-output",
            {
                'voice_speed': 0.9,
                'fade_duration': 3.0,
                'bell_volume': 0.6,
                'pause_before_bell': 2.0,
                'pause_after_bell': 2.0
            }
        )
        if not result or 'voice' not in result or not os.path.isfile(result['voice']):
            return {'error': 'Failed to stitch final audio'}, 500
        final_file = result['voice']

        #khởi tạo supabase và xử lý upload lên supabase
        supabase_service = SupabaseService()
    
        with open(final_file, 'rb') as f:
            file_data = f.read()
        
        file_name = f"{int(time.time())}-{os.path.basename(final_file).replace(' ', '_')}"

        voice_upload = supabase_service.upload_file(file_data, file_name)
        if not voice_upload['success']:
            return {
                "status": "error",
                "stage": "upload_voice",
                "http_code": voice_upload['http_code'],
                "response": voice_upload['response']
            }, 500

        insert_data = {
            "script": request.form.get('script'),
            "audio_url": f"/{supabase_service.bucket}/{file_name}",
            "background_audio_id": request.form.get('bg_audio'),
            "status": "completed",
            "questionnaire_data": {},
            "user_id": None
        }

        db_result = supabase_service.insert_audio_session(insert_data)
        
        delete_folder(project_path)
        
        if db_result['success']:
            return {
                "status": "ok",
                "voice_file": file_name,
                "db_response": db_result['response']
            }
        else:
            return {
                "status": "error",
                "stage": "db_insert",
                "http_code": db_result['http_code'],
                "response": db_result['response']
            }, 500
            
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return {
            'error': 'Server Error',
            'message': str(e)
        }, 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/api/process', methods=['POST'])
def process():
    result, status_code = process_audio_request()
    return jsonify(result), status_code

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Flask API Server - Python equivalent of PHP /api/process',
        'endpoints': {
            '/health': 'Health check',
            '/api/process': 'Main audio processing endpoint (POST)'
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
