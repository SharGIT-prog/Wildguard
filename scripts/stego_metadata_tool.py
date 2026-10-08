import os
import io
import time
import uuid
import random
import struct
import zlib
import datetime
import http.server
import socketserver
import urllib.parse
from PIL import Image, PngImagePlugin
import piexif

PORT = 8080

# --- Random Data Pools for Unique Metadata & Naming ---
ANIMALS = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']
FORESTS = ["BlackForest", "RedwoodGrove", "Tongass", "AmazonBasin", "Bialowieza", "Daintree", "Monteverde", "Tarkine"]
CITIES = ["Kyoto", "Reykjavik", "Vancouver", "Innsbruck", "Queenstown", "Bergen", "Valdivia", "Kathmandu"]

CAMERAS = [
    ("Sony", "ILCE-7RM5", "FE 24-70mm F2.8 GM II"),
    ("Canon", "EOS R5", "RF 50mm F1.2 L USM"),
    ("Nikon", "Z8", "NIKKOR Z 24-120mm f/4 S"),
    ("Fujifilm", "X-T5", "XF 16-55mm F2.8 R LM WR"),
    ("Leica", "M11", "Summicron-M 35mm f/2 ASPH")
]

CREATORS = ["Elena Rostova", "Marcus Vance", "Ron Weasley", "Freja Lind", "Kenji Sato", "Amara Okafor"]

def deg_to_dms_rational(deg_float):
    deg = int(abs(deg_float))
    min_float = (abs(deg_float) - deg) * 60.0
    minute = int(min_float)
    sec_float = (min_float - minute) * 60.0
    sec = int(round(sec_float * 1000))
    return ((deg, 1), (minute, 1), (sec, 1000))

def generate_unique_metadata():
    rand_id = uuid.uuid4().hex[:8]
    camera_make, camera_model, lens_model = random.choice(CAMERAS)
    creator = random.choice(CREATORS)
    animal = random.choice(ANIMALS)
    forest = random.choice(FORESTS)
    city = random.choice(CITIES)
    
    # Coordinates & Time
    lat = round(random.uniform(-60.0, 70.0), 6)
    lon = round(random.uniform(-170.0, 170.0), 6)
    alt = round(random.uniform(5.0, 4200.0), 2)
    
    random_days_ago = random.randint(1, 3650)
    random_seconds = random.randint(0, 86400)
    dt = datetime.datetime.now() - datetime.timedelta(days=random_days_ago, seconds=random_seconds)
    time_str = dt.strftime("%Y:%m:%d %H:%M:%S")
    date_gps_str = dt.strftime("%Y:%m:%d")
    
    return {
        "uid": rand_id,
        "filename": f"{animal}_{forest}_{city}_{rand_id}",
        "make": camera_make,
        "model": camera_model,
        "lens": lens_model,
        "serial": f"SN-{random.randint(1000000, 9999999)}",
        "lens_serial": f"LNS-{random.randint(1000000, 9999999)}",
        "creator": creator,
        "location": f"{forest}, near {city}",
        "description": f"Field observation of {animal} in {forest} region. Node ID: {rand_id}",
        "keywords": f"{animal}, {forest}, {city}, Wildlife, Survey-{rand_id}",
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "datetime_str": time_str,
        "gps_date_str": date_gps_str,
        "time_obj": dt
    }

def inject_lsb_variance(img: Image.Image, payload_entropy: bytes) -> Image.Image:
    """Injects high-entropy steganographic bit patterns into the least significant bit plane."""
    img = img.convert("RGBA" if img.mode == "RGBA" else "RGB")
    pixels = bytearray(img.tobytes())
    payload_len = len(payload_entropy)
    
    if payload_len == 0:
        return img

    for i in range(min(len(pixels), payload_len * 8)):
        byte_idx = i // 8
        bit_idx = i % 8
        payload_bit = (payload_entropy[byte_idx] >> bit_idx) & 1
        pixels[i] = (pixels[i] & ~1) | payload_bit
        
    return Image.frombytes(img.mode, img.size, bytes(pixels))

def process_jpeg(image: Image.Image, meta: dict, raw_payload: bytes) -> bytes:
    # 1. EXIF and GPS Dictionary assembly
    lat_dms = deg_to_dms_rational(meta["lat"])
    lon_dms = deg_to_dms_rational(meta["lon"])
    
    gps_dict = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: "N" if meta["lat"] >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: lat_dms,
        piexif.GPSIFD.GPSLongitudeRef: "E" if meta["lon"] >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: lon_dms,
        piexif.GPSIFD.GPSAltitudeRef: 0,
        piexif.GPSIFD.GPSAltitude: (int(meta["alt"] * 100), 100),
        piexif.GPSIFD.GPSTimeStamp: ((meta["time_obj"].hour, 1), (meta["time_obj"].minute, 1), (meta["time_obj"].second, 1)),
        piexif.GPSIFD.GPSDateStamp: meta["gps_date_str"].encode("utf-8")
    }
    
    zeroth_dict = {
        piexif.ImageIFD.Make: meta["make"].encode("utf-8"),
        piexif.ImageIFD.Model: meta["model"].encode("utf-8"),
        piexif.ImageIFD.Software: f"Firmware v{random.randint(1,9)}.{random.randint(0,9)}.0".encode("utf-8"),
        piexif.ImageIFD.DateTime: meta["datetime_str"].encode("utf-8"),
        piexif.ImageIFD.Artist: meta["creator"].encode("utf-8"),
        piexif.ImageIFD.ImageDescription: meta["description"].encode("utf-8"),
    }
    
    exif_dict = {
        piexif.ExifIFD.DateTimeOriginal: meta["datetime_str"].encode("utf-8"),
        piexif.ExifIFD.DateTimeDigitized: meta["datetime_str"].encode("utf-8"),
        piexif.ExifIFD.BodySerialNumber: meta["serial"].encode("utf-8"),
        piexif.ExifIFD.LensSerialNumber: meta["lens_serial"].encode("utf-8"),
        piexif.ExifIFD.LensModel: meta["lens"].encode("utf-8"),
        piexif.ExifIFD.UserComment: f"IPTC-Keywords: {meta['keywords']}".encode("utf-8")
    }
    
    # 2. Embedded Thumbnail Generation
    thumb = image.copy()
    thumb.thumbnail((160, 120))
    thumb_buf = io.BytesIO()
    thumb.convert("RGB").save(thumb_buf, format="JPEG", quality=75)
    
    exif_bytes = piexif.dump({
        "0th": zeroth_dict,
        "Exif": exif_dict,
        "GPS": gps_dict,
        "1st": {},
        "thumbnail": thumb_buf.getvalue()
    })
    
    # 3. LSB injection
    mod_img = inject_lsb_variance(image, raw_payload)
    out_buf = io.BytesIO()
    mod_img.convert("RGB").save(out_buf, format="JPEG", quality=95, exif=exif_bytes)
    jpeg_bytes = bytearray(out_buf.getvalue())
    
    # 4. Insert custom APP13 (IPTC block simulation) segment before SOS
    app13_marker = b"\xFF\xED"
    app13_payload = f"Photoshop 3.0\x008BIM\x04\x04\x00\x00\x00\x00\x00\x1a\x1c\x02\x78\x00\x14{meta['caption_abstract'] if 'caption_abstract' in meta else meta['description'][:20]}".encode("utf-8", "ignore")
    app13_len = struct.pack(">H", len(app13_payload) + 2)
    app13_chunk = app13_marker + app13_len + app13_payload
    
    # Insert right after SOI (FF D8)
    jpeg_bytes = jpeg_bytes[:2] + app13_chunk + jpeg_bytes[2:]
    
    # 5. Steganographic Appended Overlay past EOF (FF D9)
    overlay_payload = b"\r\n--- COVERT CHANNEL BUFFER BEGIN ---\r\n" + raw_payload + b"\r\n--- COVERT CHANNEL BUFFER END ---\r\n"
    return bytes(jpeg_bytes) + overlay_payload

def process_png(image: Image.Image, meta: dict, raw_payload: bytes) -> bytes:
    # 1. Custom PNG Chunks & Text Chunks (tEXt, zTXt, iTXt)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Make", meta["make"])
    pnginfo.add_text("Model", meta["model"])
    pnginfo.add_text("Serial", meta["serial"])
    pnginfo.add_text("LensSerialNumber", meta["lens_serial"])
    pnginfo.add_text("Author", meta["creator"])
    pnginfo.add_text("Description", meta["description"])
    pnginfo.add_text("Keywords", meta["keywords"])
    pnginfo.add_text("GPSCoordinates", f"{meta['lat']}, {meta['lon']}, Alt: {meta['alt']}m")
    pnginfo.add_text("DateTimeOriginal", meta["datetime_str"])
    pnginfo.add_itxt("Location", meta["location"], lang="en", tkey="Location")
    
    # 2. LSB Variance injection in pixel bytes
    mod_img = inject_lsb_variance(image, raw_payload)
    out_buf = io.BytesIO()
    mod_img.save(out_buf, format="PNG", pnginfo=pnginfo)
    png_bytes = out_buf.getvalue()
    
    # 3. Inject custom private chunk (`coVt`) before IEND
    iend_pos = png_bytes.rfind(b"IEND")
    if iend_pos != -1:
        chunk_type = b"coVt"
        chunk_data = zlib.compress(raw_payload)
        chunk_len = struct.pack(">I", len(chunk_data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xffffffff)
        custom_chunk = chunk_len + chunk_type + chunk_data + crc
        
        # Split right before length header of IEND (4 bytes before "IEND")
        insert_idx = iend_pos - 4
        png_bytes = png_bytes[:insert_idx] + custom_chunk + png_bytes[insert_idx:]
    
    # 4. Trailing overlay past PNG EOF (after IEND CRC)
    overlay_payload = b"\x00\xDE\xAD\xBE\xEF" + raw_payload
    return png_bytes + overlay_payload

# --- Web Interface & Server ---
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Stego & Metadata Injection Engine</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; display: flex; flex-direction: column; align-items: center; }
        .card { background: #1e293b; border-radius: 12px; padding: 30px; width: 100%; max-width: 680px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }
        h1 { margin-top: 0; font-size: 24px; color: #38bdf8; text-align: center; }
        p { color: #94a3b8; font-size: 14px; line-height: 1.5; text-align: center; }
        #drop-zone { border: 2px dashed #38bdf8; border-radius: 8px; padding: 40px 20px; text-align: center; cursor: pointer; transition: 0.2s; background: rgba(56, 189, 248, 0.05); margin-top: 20px; }
        #drop-zone.hover { background: rgba(56, 189, 248, 0.15); border-color: #7dd3fc; }
        #file-input { display: none; }
        .log-box { margin-top: 25px; background: #020617; border-radius: 6px; padding: 15px; font-family: monospace; font-size: 12px; color: #a5f3fc; max-height: 250px; overflow-y: auto; border: 1px solid #1e293b; }
        .btn { display: inline-block; margin-top: 15px; background: #38bdf8; color: #0f172a; font-weight: bold; padding: 10px 20px; border-radius: 6px; text-decoration: none; border: none; cursor: pointer; }
        .btn:hover { background: #7dd3fc; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Stego & Metadata Injection Engine</h1>
        <p>Drop a <strong>JPEG</strong> or <strong>PNG</strong> image to generate non-repeating EXIF/GPS/XMP/IPTC headers, LSB spatial variance, custom PNG/APP segments, and trailing EOF markers.</p>
        
        <div id="drop-zone">
            <span>Drag & drop image here or <strong style="color: #38bdf8;">Browse</strong></span>
            <input type="file" id="file-input" accept="image/jpeg, image/png">
        </div>
        
        <div class="log-box" id="log">Ready for file drop...</div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const logBox = document.getElementById('log');

        function appendLog(msg) {
            logBox.innerText += '\\n' + msg;
            logBox.scrollTop = logBox.scrollHeight;
        }

        dropZone.onclick = () => fileInput.click();
        
        ['dragenter', 'dragover'].forEach(e => {
            dropZone.addEventListener(e, (evt) => { evt.preventDefault(); dropZone.classList.add('hover'); });
        });
        
        ['dragleave', 'drop'].forEach(e => {
            dropZone.addEventListener(e, (evt) => { evt.preventDefault(); dropZone.classList.remove('hover'); });
        });

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length) uploadFile(files[0]);
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) uploadFile(fileInput.files[0]);
        });

        function uploadFile(file) {
            appendLog(`Processing: ${file.name} (${(file.size/1024).toFixed(1)} KB)...`);
            const formData = new FormData();
            formData.append('image', file);

            fetch('/process', { method: 'POST', body: formData })
            .then(res => {
                const disp = res.headers.get('Content-Disposition');
                let filename = 'processed_image';
                if (disp && disp.includes('filename=')) {
                    filename = disp.split('filename=')[1].replace(/["']/g, '').trim();
                }
                return res.blob().then(blob => ({ blob, filename }));
            })
            .then(({ blob, filename }) => {
                appendLog(`Success! Injected metadata, LSB entropy, and covert chunks.`);
                appendLog(`Triggering download as: ${filename}`);
                
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
            })
            .catch(err => {
                appendLog(`Error processing image: ${err}`);
            });
        }
    </script>
</body>
</html>
"""

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/process":
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self.send_error(400, "Bad Request: Expected multipart/form-data")
                return

            boundary = content_type.split("boundary=")[1].strip()
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)

            # Extract image raw bytes from multipart body
            boundary_bytes = ("--" + boundary).encode("utf-8")
            parts = body.split(boundary_bytes)
            
            file_data = None
            orig_name = "image.jpg"
            for part in parts:
                if b'name="image"' in part:
                    header_end = part.find(b"\r\n\r\n")
                    if header_end != -1:
                        raw_headers = part[:header_end].decode("utf-8", "ignore")
                        if 'filename="' in raw_headers:
                            orig_name = raw_headers.split('filename="')[1].split('"')[0]
                        file_data = part[header_end + 4:].rstrip(b"\r\n--")
                        break

            if not file_data:
                self.send_error(400, "Could not parse image")
                return

            try:
                img = Image.open(io.BytesIO(file_data))
                img_format = img.format
                meta = generate_unique_metadata()
                
                # Dynamic cryptographic entropy seed for covert payload
                raw_payload = f"UID:{meta['uid']}|TS:{time.time()}|HASH:{uuid.uuid4()}|COORD:{meta['lat']},{meta['lon']}".encode("utf-8")

                if img_format == "PNG":
                    out_bytes = process_png(img, meta, raw_payload)
                    ext = ".png"
                    mime = "image/png"
                else:
                    out_bytes = process_jpeg(img, meta, raw_payload)
                    ext = ".jpg"
                    mime = "image/jpeg"

                final_name = f"{meta['filename']}{ext}"
                
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Disposition", f'attachment; filename="{final_name}"')
                self.send_header("Content-Length", str(len(out_bytes)))
                self.end_headers()
                self.wfile.write(out_bytes)

            except Exception as e:
                self.send_error(500, f"Processing error: {str(e)}")
        else:
            self.send_error(404, "Not Found")

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        print(f"[*] Engine running on http://localhost:{PORT}")
        print("[*] Drag and drop an image via your browser.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
