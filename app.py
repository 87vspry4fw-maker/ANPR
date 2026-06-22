from pathlib import Path

from flask import ( Flask,
                    render_template,
                    request,
                    redirect,
                    url_for,
                    flash,)

from werkzeug.utils import secure_filename

from prediction import PredictPlate
from database import CarParkDB

BaseDir = Path(__file__).resolve().parent
UploadDir = BaseDir / "uploads"
UploadDir.mkdir(exist_ok=True)
AllowedExtensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

app = Flask(__name__)
app.secret_key = "change-me-for-the-nea"
db = CarParkDB(str(BaseDir / "ANPR.db"))

def _normalise(plate):
    return plate.replace(" ", "").upper()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please choose an image first.")
        return redirect(url_for("index"))
    
    ext = Path(file.filename).suffix.lower()
    if ext not in AllowedExtensions:
        flash(f"Unsupported file type '{ext}'. Use a PNG or a JPEG.")
        return redirect(url_for("index"))
    
    save_path = UploadDir / secure_filename(file.filename)
    file.save(save_path)

    plate = PredictPlate(save_path)

    if not plate:
        flash("No characters could be read. Crop the photo to just the plate and try again.")
        return redirect(url_for("index"))
    
    return redirect(url_for("result", plate=plate))

@app.route("/result")
def result():
    plate = _normalise(request.args.get("plate", ""))
    if not plate:
        return redirect(url_for("index"))
    return render_template("result.html", plate=plate, is_banned=db.is_banned(plate))

@app.route("/details/<plate>")
def details(plate):
    plate = _normalise(plate)
    record = db.get_details(plate)
    if not record:
        flash(f"No record found for plate '{plate}'.")
        return redirect(url_for("index"))
    return render_template("details.html", plate=plate, record=record, history=db.get_history(plate))

@app.route("/ban/<plate>", methods=["GET", "POST"])
def ban(plate):
    if request.method == "GET":
        plate = _normalise(plate)
        return render_template("ban.html", plate=plate)
    
    plate = _normalise(plate)
    reason = request.form.get("reason", "").strip()
    student_name = request.form.get("student_name", "").strip()
    student_id = request.form.get("student_id", "").strip()
    if not reason or not student_name or not student_id:
        flash("All fields are required to ban a vehicle.")
        return redirect(url_for("ban", plate=plate))
    
    db.ban(plate, reason, student_name, student_id)
    flash(f"{plate} has been banned.")
    return redirect(url_for("details", plate=plate))

@app.route("/unban/<plate>", methods=["POST"])
def unban(plate):
    plate = _normalise(plate)
    db.unban(plate)
    flash(f"{plate} has been unbanned.")
    return redirect(url_for("details", plate=plate))

@app.route("/history")
def history():
    return render_template("history.html", events=db.recent_events(limit=50))

if __name__ == "__main__":
    app.run(debug=True)
