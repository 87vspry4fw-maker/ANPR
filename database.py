import sqlite3
DB = "anpr.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS banned_vehicles
                 (Plate TEXT PRIMARY KEY,
                  Reason TEXT NOT NULL,
                  Student_Name TEXT NOT NULL,
                  Student_ID INTEGER NOT NULL,
                  Date_Banned TEXT NOT NULL
                  )''')
    conn.commit()
    conn.close()

def normalise(plate):
    return plate.upper().replace(" ", "")

def get_ban(plate):
    plate = normalise(plate)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM banned_vehicles WHERE Plate=?", (plate,))
    result = c.fetchone()
    conn.close()
    return result

def add_ban(plate, reason, student_name, student_id, date_banned):
    plate = normalise(plate)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO banned_vehicles (Plate, Reason, Student_Name, Student_ID, Date_Banned) VALUES (?, ?, ?, ?, ?)",
              (plate, reason, student_name, student_id, date_banned))
    conn.commit()
    conn.close()