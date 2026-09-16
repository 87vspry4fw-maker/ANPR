import sqlite3
from datetime import datetime 

class CarParkDB:
    def __init__(self, db_path="ANPR.db"):
        self.db_path = db_path
        self.init_DB()

    #Helpers
    @staticmethod
    def _normalise(plate):
        return plate.replace(" ", "").upper()

    @staticmethod
    def _now():
        return datetime.now().isoformat()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    #Databases
    def init_DB(self):
        conn = self._get_connection()
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
                    plate TEXT NOT NULL,
                    action TEXT NOT NULL,   -- 'ban' or 'unban'
                    reason TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (plate) REFERENCES vehicles(plate)
                )
            """)
        conn.close()

    #Operations
    def is_banned(self, plate):
        conn = self._get_connection()
        plate = self._normalise(plate)
        row = conn.execute("SELECT is_banned FROM vehicles WHERE plate = ?", (plate,)).fetchone()
        conn.close()
        return bool(row and row["is_banned"])

    def ban(self, plate, reason, student_name=None, student_id=None):
        plate = self._normalise(plate)
        now = self._now()
        conn = self._get_connection()
        with conn:
            conn.execute("""
                INSERT INTO vehicles (plate, is_banned, student_name, student_id, reason, updated_at)
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(plate) DO UPDATE SET
                    is_banned = 1,
                    reason = ?,
                    updated_at = ?
            """, (plate, student_name, student_id, reason, now, reason, now))
            conn.execute("""
                INSERT INTO ban_events (plate, action, reason, timestamp)
                VALUES (?, 'ban', ?, ?)
            """, (plate, reason, now))
        conn.close()

    def unban(self, plate):
        plate = self._normalise(plate)
        now = self._now()
        conn = self._get_connection()
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

    def get_details(self, plate):
        conn = self._get_connection()
        plate = self._normalise(plate)
        row = conn.execute("SELECT * FROM vehicles WHERE plate = ?", (plate,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_vehicle_details(self):
        #gets all vehicles in the database, ordered by plates
        conn = self._get_connection()
        rows = conn.execute("SELECT plate, is_banned, student_name, student_id, updated_at FROM vehicles ORDER BY plate").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_history(self, plate):
        #kind of realising now I mislabelled this, would be history for a specific plate
        conn = self._get_connection()
        plate = self._normalise(plate)
        rows = conn.execute("SELECT action, reason, timestamp FROM ban_events WHERE plate = ? ORDER BY timestamp DESC", (plate,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def recent_events(self, limit=50):
        #DB-wide history (every plate), most recent first - the history tab
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT plate, action, reason, timestamp FROM ban_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]