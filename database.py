import sqlite3
from datetime import datetime, timezone
DB = "ANPR.db"

#Helpers
def Normalise(plate):
    return plate.replace(" ", "").upper()

def Now():
    return datetime.now().isoformat()

def GetConnection():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

#Databases
def initDB():
    conn = GetConnection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                plate TEXT PRIMARY KEY,
                is_banned INTEGER NOT NULL DEFAULT 0,
                student_name TEXT,
                student_id INTEGER,
                reason TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ban_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                action TEXT NOT NULL,   -- 'ban' or 'unban'
                reason TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (plate) REFERENCES vehicles(plate)
            )
        """)
    conn.close()

#Operations
def is_banned(plate):
    conn = GetConnection()
    plate = Normalise(plate)
    row = conn.execute("SELECT is_banned FROM vehicles WHERE plate = ?", (plate,)).fetchone()
    conn.close()
    return bool(row and row["is_banned"])

def ban(plate, reason):
    plate = Normalise(plate)
    now = Now()
    conn = GetConnection()
    with conn:
        conn.execute("""
            INSERT INTO vehicles (plate, is_banned, reason, updated_at)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(plate) DO UPDATE SET
                is_banned = 1,
                reason = ?,
                updated_at = ?
        """, (plate, reason, now, reason, now))
        conn.execute("""
            INSERT INTO ban_events (plate, action, reason, timestamp)
            VALUES (?, 'ban', ?, ?)
        """, (plate, reason, now))
    conn.close()

def unban(plate):
    plate = Normalise(plate)
    now = Now()
    conn = GetConnection()
    with conn:
        conn.execute("""
            UPDATE vehicles SET is_banned = 0, updated_at = ?
            WHERE plate = ?
        """, (now, plate))
        conn.execute("""
            INSERT INTO ban_events (plate, action, timestamp)
            VALUES (?, 'unban', ?)
        """, (plate, now))
    conn.close()

def get_details(plate):
    conn = GetConnection()
    plate = Normalise(plate)
    row = conn.execute("SELECT * FROM vehicles WHERE plate = ?", (plate,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_history(plate):
    conn = GetConnection()
    plate = Normalise(plate)
    rows = conn.execute("SELECT action, reason, timestamp FROM ban_events WHERE plate = ? ORDER BY timestamp DESC", (plate,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

#Example usage
if __name__ == "__main__":
    initDB()
    plate = "AB12 CDE"
    print("Initial state:", get_details(plate))

    ban(plate, reason="No permit")
    print("After ban:  ", get_details(plate))
    print("Is banned?  ", is_banned(plate))

    unban(plate)
    print("After unban:  ", get_details(plate))
    print("Is banned?  ", is_banned(plate))

    ban(plate, reason="Repeat offender")
    print("After second ban:  ", get_details(plate))

    print("History:")
    for event in get_history(plate):
        print("  ", event)

