from pathlib import Path

from flask import ( Flask,
                    render_template,
                    request,
                    redirect,
                    url_for,
                    session,
                    flash,)

from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
from functools import wraps


from prediction import PredictPlate
from database import CarParkDB

BaseDir = Path(__file__).resolve().parent
UploadDir = BaseDir / "uploads"
UploadDir.mkdir(exist_ok=True)
AllowedExtensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

app = Flask(__name__)
app.secret_key = "change-me-for-the-nea"
Username = "admin"
Password_hash = "pbkdf2:sha256:1000000$mozSeLmAFHPr9Dp3$2f356340a657a8f0b911428b9953af055efee2038671f4590961c13d5b8a78fd"
#Parking@Godalming = password

db = CarParkDB(str(BaseDir / "ANPR.db"))

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

def _normalise(plate):
    return plate.replace(" ", "").upper()

@app.template_filter("format_plate")
def format_plate(plate):
    plate = plate or ""
    if len(plate) == 7:            # standard UK plate: AB12 CDE
        return plate[:4] + " " + plate[4:]
    return plate                   # leave odd-length reads untouched

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == Username and check_password_hash(Password_hash, password):
            session["logged_in"] = True
            return redirect(url_for("index"))
        flash("Incorrect username or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required
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

    plate, confidences = PredictPlate(save_path)

    if not plate:
        flash("No characters could be read. Crop the photo to just the plate and try again.")
        return redirect(url_for("index"))
    
    Threshold = 0.5
    low = [
        f"position {i+1} ('{char}', {conf:.0%})"
        for i, (char, conf) in enumerate(confidences)
        if conf < Threshold
    ]
    if low:
        flash("Low confidence on " + ", ".join(low) + " - the reading may be wrong, consider retaking the photo.")

    return redirect(url_for("result", plate=plate))

@app.route("/result")
@login_required
def result():
    plate = _normalise(request.args.get("plate", ""))
    if not plate:
        return redirect(url_for("index"))
    return render_template("result.html", plate=plate, is_banned=db.is_banned(plate))

@app.route("/details/<plate>")
@login_required
def details(plate):
    plate = _normalise(plate)
    record = db.get_details(plate)
    if not record:
        flash(f"No record found for plate '{plate}'.")
        return redirect(url_for("index"))
    return render_template("details.html", plate=plate, record=record, history=db.get_history(plate))

@app.route("/ban/<plate>", methods=["GET", "POST"])
@login_required
def ban(plate):
    plate = _normalise(plate)
    record = db.get_details(plate)
    # "known" = we already have this student's name and ID from a previous ban
    known = bool(record and record["student_name"] and record["student_id"])

    if request.method == "GET":
        return render_template("ban.html", plate=plate, record=record, known=known)

    reason = request.form.get("reason", "").strip()

    if known:
        if not reason:
            flash("A reason is required to ban a vehicle.")
            return redirect(url_for("ban", plate=plate))
        db.ban(plate, reason)            # ON CONFLICT keeps the existing name/ID
    else:
        student_name = request.form.get("student_name", "").strip()
        student_id = request.form.get("student_id", "").strip()
        if not reason or not student_name or not student_id:
            flash("All fields are required to ban a vehicle.")
            return redirect(url_for("ban", plate=plate))
        db.ban(plate, reason, student_name, student_id)

    flash(f"{plate} has been banned.")
    return redirect(url_for("details", plate=plate))

@app.route("/unban/<plate>", methods=["POST"])
@login_required
def unban(plate):
    plate = _normalise(plate)
    db.unban(plate)
    flash(f"{plate} has been unbanned.")
    return redirect(url_for("details", plate=plate))

@app.route("/history")
@login_required
def history():
    return render_template("history.html", events=db.recent_events(limit=50))

if __name__ == "__main__":
    app.run(debug=True, port=8000)
