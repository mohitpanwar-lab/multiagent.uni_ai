#!/usr/bin/env python
# coding: utf-8

"""
tools.py
Loads real student data from JSON files (student_records.json,
attendance_records.json, fee_payments.json, results_marks.json,
faculty_records.json, time_table.json, course_information.json)
and exposes @tool functions for a LangChain/LangGraph style agent.

Put all 7 JSON files in the SAME FOLDER as this tools.py file
(or set the DATA_FOLDER env var / edit DATA_FOLDER below).
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


class SimpleTool:
    """
    Lightweight stand-in for langchain's @tool decorator.
    Wraps a plain function so it can be called either directly
    (my_func(...)) or the langchain way (my_func.invoke({...})).
    """

    def __init__(self, func):
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def invoke(self, kwargs):
        return self.func(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def tool(func):
    return SimpleTool(func)


tavily_client = None


# ---------------------------------------------------------------------------
# Data loading — real JSON files
# ---------------------------------------------------------------------------

# Folder where the 7 JSON files live. Defaults to the same folder as this
# tools.py file. Override with the DATA_FOLDER env var if you keep the
# JSON files in a separate "data" folder instead.
DATA_FOLDER = Path(os.getenv("DATA_FOLDER", Path(__file__).resolve().parent))


def _load_json(filename: str):
    path = DATA_FOLDER / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _index_by(records, key):
    """Build a {UPPER_KEY_VALUE: record} lookup dict, skipping bad rows."""
    index = {}
    for r in records:
        val = r.get(key)
        if val:
            index[str(val).upper()] = r
    return index


def _load_json_safe(filename: str, default):
    """
    Load one JSON file, falling back to `default` (and printing a warning)
    if the file is missing OR malformed — so one bad/missing file doesn't
    take down every tool, only the ones that depend on it.
    """
    try:
        return _load_json(filename)
    except FileNotFoundError:
        print(f"[tools.py] Warning: '{filename}' not found in {DATA_FOLDER} — related tools will report no data.")
        return default
    except json.JSONDecodeError as e:
        print(f"[tools.py] Warning: '{filename}' has invalid JSON ({e}) — related tools will report no data.")
        return default


STUDENT_RECORDS = _load_json_safe("student_records.json", [])
ATTENDANCE_RECORDS = _load_json_safe("attendance_records.json", [])
FEE_PAYMENTS = _load_json_safe("fee_payments.json", [])
RESULTS_MARKS = _load_json_safe("results_marks.json", [])
FACULTY_RECORDS = _load_json_safe("faculty_records.json", [])
TIME_TABLE = _load_json_safe("time_table.json", {})
COURSE_INFORMATION = _load_json_safe("course_information.json", {})
LIBRARY_RECORDS = _load_json_safe("library_records.json", [])

# Fast lookup indexes — every student can be found by EITHER their
# student_id (e.g. "STU2026001") or their roll_no (e.g. "26CSE001").
STUDENT_BY_ID = _index_by(STUDENT_RECORDS, "student_id")
STUDENT_BY_ROLL = _index_by(STUDENT_RECORDS, "roll_no")
ATTENDANCE_BY_ID = _index_by(ATTENDANCE_RECORDS, "student_id")
FEE_BY_ID = _index_by(FEE_PAYMENTS, "student_id")
RESULTS_BY_ID = _index_by(RESULTS_MARKS, "student_id")
LIBRARY_BY_ID = _index_by(LIBRARY_RECORDS, "student_id")


def _resolve_student(student_id: str):
    """Accepts either a student_id ('STU2026001') or a roll_no ('26CSE001')."""
    key = student_id.strip().upper()
    return STUDENT_BY_ID.get(key) or STUDENT_BY_ROLL.get(key)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def student_profile(student_id: str) -> str:
    """
    Retrieve the complete profile of a student using their Student ID or Roll Number.
    """
    student = _resolve_student(student_id)
    if not student:
        return f"No student found with ID/Roll No '{student_id}'."

    return f"""
Student Profile

Name        : {student['name']}
Student ID  : {student['student_id']}
Roll No     : {student['roll_no']}
Branch      : {student['branch']}
Semester    : {student['semester']}
Email       : {student['email']}
Phone       : {student['phone']}
Mentor      : {student['mentor']}
"""


@tool
def student_attendance(student_id: str) -> str:
    """
    Retrieve a student's overall and subject-wise attendance using their
    Student ID or Roll Number.
    """
    student = _resolve_student(student_id)
    if not student:
        return f"No student found with ID/Roll No '{student_id}'."

    record = ATTENDANCE_BY_ID.get(student["student_id"].upper())
    if not record:
        return f"No attendance record found for '{student['name']}'."

    response = (
        f"ATTENDANCE RECORD\n\n"
        f"Name              : {student['name']}\n"
        f"Roll No           : {student['roll_no']}\n"
        f"Total Attendance  : {record['total_attendance_percentage']}%\n"
        f"Shortage Status   : {record['shortage_status']}\n\n"
        f"Subject-wise Attendance\n"
    )
    for subject, pct in record.get("subject_wise", {}).items():
        response += f"{subject:<20}: {pct}\n"

    return response


@tool
def student_marks_result(student_id: str) -> str:
    """
    Retrieve a student's internal assessment marks, previous SGPA, and
    result status using their Student ID or Roll Number.
    """
    student = _resolve_student(student_id)
    if not student:
        return f"No student found with ID/Roll No '{student_id}'."

    record = RESULTS_BY_ID.get(student["student_id"].upper())
    if not record:
        return f"No academic record found for '{student['name']}'."

    response = (
        f"STUDENT ACADEMIC RECORD\n\n"
        f"Name                  : {student['name']}\n"
        f"Roll No               : {student['roll_no']}\n"
        f"Current Semester      : {record['current_semester']}\n"
        f"Previous Semester GPA : {record['previous_semester_gpa']}\n\n"
        f"Internal Assessment Marks\n"
    )
    for subject, marks in record.get("internal_assessment_marks", {}).items():
        response += f"{subject:<20}: {marks}\n"

    response += f"\nResult Status : {record['result_status']}"
    return response


@tool
def fee_details(student_id: str) -> str:
    """
    Retrieve the fee payment details of a student using their Student ID
    or Roll Number.
    """
    student = _resolve_student(student_id)
    if not student:
        return f"No student found with ID/Roll No '{student_id}'."

    record = FEE_BY_ID.get(student["student_id"].upper())
    if not record:
        return f"No fee record found for '{student['name']}'."

    return (
        f"FEE DETAILS\n\n"
        f"Name             : {student['name']}\n"
        f"Roll No          : {student['roll_no']}\n"
        f"Semester         : {record['semester']}\n\n"
        f"Total Fee        : ₹{record['total_fees']:,}\n"
        f"Amount Paid      : ₹{record['amount_paid']:,}\n"
        f"Amount Due       : ₹{record['amount_due']:,}\n"
        f"Payment Status   : {record['payment_status']}\n"
        f"Last Payment Date: {record['last_payment_date']}\n"
        f"Receipt No       : {record['receipt_no']}\n"
    )


@tool
def class_timetable(student_id: str) -> str:
    """
    Retrieve the weekly class timetable for a student's branch/semester/section
    using their Student ID or Roll Number.
    """
    student = _resolve_student(student_id)
    if not student:
        return f"No student found with ID/Roll No '{student_id}'."

    if not TIME_TABLE:
        return "No timetable data available."

    response = (
        f"WEEKLY TIMETABLE — {TIME_TABLE.get('branch')} "
        f"Semester {TIME_TABLE.get('semester')} Section {TIME_TABLE.get('section')}\n\n"
    )
    for day, classes in TIME_TABLE.get("weekly_schedule", {}).items():
        response += f"{day}:\n"
        for c in classes:
            response += f"  {c['time']} — {c['subject']} ({c['room']})\n"
        response += "\n"

    return response


@tool
def faculty_information(name_or_department: str) -> str:
    """
    Retrieve faculty information by (partial) name or department.
    """
    query = name_or_department.strip().lower()
    matches = [
        f for f in FACULTY_RECORDS
        if query in f["name"].lower() or query in f["department"].lower()
    ]

    if not matches:
        return f"No faculty found matching '{name_or_department}'."

    response = ""
    for f in matches:
        response += (
            f"Faculty Name  : {f['name']}\n"
            f"Designation   : {f['designation']}\n"
            f"Department    : {f['department']}\n"
            f"Cabin No      : {f['cabin_no']}\n"
            f"Email         : {f['email']}\n"
            f"Office Hours  : {f['office_hours']}\n\n"
        )
    return response.strip()


@tool
def course_subjects(student_id: str) -> str:
    """
    Retrieve the course/subject details for a student's semester using
    their Student ID or Roll Number.
    """
    student = _resolve_student(student_id)
    if not student:
        return f"No student found with ID/Roll No '{student_id}'."

    if not COURSE_INFORMATION:
        return "No course information available."

    sem = student["semester"]
    key = f"semester_{sem}_courses"
    courses = COURSE_INFORMATION.get(key, [])

    response = (
        f"COURSE DETAILS\n"
        f"Student Name : {student['name']}\n"
        f"Roll No      : {student['roll_no']}\n"
        f"Branch       : {student['branch']}\n"
        f"Semester     : {sem}\n\n"
        f"Subjects:\n"
    )
    for i, c in enumerate(courses, start=1):
        response += (
            f"{i}. {c['course_name']} ({c['course_code']}) — "
            f"{c['credits']} credits\n"
        )

    return response


@tool
def university_notices() -> str:
    """Retrieve the latest university notices and announcements."""
    return """
University Notices

1. Mid Semester Exams begin from 20 August 2026.
2. Fee submission deadline is 31 August 2026.
3. Independence Day celebration on 15 August.
4. Placement Registration closes on 25 August.
5. Library will remain closed on Sunday.
"""


@tool
def library_details(student_id: str) -> str:
    """
    Retrieve a student's library record — books issued and pending fine —
    using their Student ID or Roll Number.
    """
    student = _resolve_student(student_id)
    if not student:
        return f"No student found with ID/Roll No '{student_id}'."

    record = LIBRARY_BY_ID.get(student["student_id"].upper())
    if not record:
        return f"No library record found for '{student['name']}'."

    response = (
        f"LIBRARY DETAILS\n\n"
        f"Name             : {student['name']}\n"
        f"Roll No          : {student['roll_no']}\n"
        f"Membership       : {record['membership_status']}\n"
        f"Books Issued     : {record['books_issued']}\n"
    )
    if record["books"]:
        response += "\nBooks:\n"
        for b in record["books"]:
            response += f"- {b}\n"
    response += f"\nLibrary Fine     : ₹{record['fine']}"
    return response


UNIVERSITY_POLICIES = {
    "attendance": (
        "Attendance Policy\n"
        "- Minimum 75% attendance is mandatory in every subject to be eligible "
        "for the semester-end examination.\n"
        "- Students below 75% but above 65% may apply for condonation with a "
        "valid medical/genuine reason, subject to HOD approval.\n"
        "- Below 65% attendance leads to detention from the exam."
    ),
    "exam": (
        "Examination Policy\n"
        "- Minimum 75% attendance and clearance of all dues is required to "
        "sit for semester-end exams.\n"
        "- A student must score at least 40% (internal + external combined) "
        "to pass a subject.\n"
        "- Re-evaluation requests must be submitted within 7 days of result "
        "declaration."
    ),
    "fee": (
        "Fee & Refund Policy\n"
        "- Semester fee must be paid before the due date shown in your fee "
        "details; a late fee applies after the due date.\n"
        "- Fees are refundable only if withdrawal is requested within 7 days "
        "of the start of the semester, as per UGC refund rules.\n"
        "- No refund is applicable after the semester has commenced beyond "
        "7 days."
    ),
    "leave": (
        "Leave Policy\n"
        "- Leave requests must be submitted in advance except in medical "
        "emergencies.\n"
        "- Leave exceeding 7 consecutive days requires HOD approval and "
        "supporting documents.\n"
        "- Unapproved absence affects attendance eligibility."
    ),
    "conduct": (
        "Code of Conduct\n"
        "- Ragging in any form is strictly prohibited and is a punishable "
        "offence under university and UGC regulations.\n"
        "- Use of unfair means in examinations leads to disciplinary action, "
        "including possible debarment.\n"
        "- Students are expected to maintain discipline in classrooms, labs, "
        "and hostel premises."
    ),
    "grading": (
        "Grading Policy\n"
        "- The university follows a 10-point CGPA grading system.\n"
        "- Grades are awarded based on combined internal assessment and "
        "semester-end examination performance.\n"
        "- A minimum SGPA of 5.0 is required to be promoted without backlog."
    ),
}


@tool
def university_policy(topic: str = "") -> str:
    """
    Retrieve official university policy information. Pass a topic such as
    'attendance', 'exam', 'fee', 'leave', 'conduct', or 'grading' to get
    that specific policy, or leave it empty to get all policies.
    """
    topic_key = topic.strip().lower()

    if not topic_key:
        return "\n\n".join(UNIVERSITY_POLICIES.values())

    for key, text in UNIVERSITY_POLICIES.items():
        if topic_key in key or key in topic_key:
            return text

    return (
        f"No specific policy found for '{topic}'. Available policy topics: "
        + ", ".join(UNIVERSITY_POLICIES.keys())
    )


@tool
def submit_leave(student_name: str, reason: str, days: int) -> str:
    """
    Submit a leave request.
    """
    return (
        f"Leave Request Submitted Successfully.\n\n"
        f"Student : {student_name}\n"
        f"Reason  : {reason}\n"
        f"Days    : {days}\n"
        f"Status  : Pending Approval"
    )


@tool
def submit_complaint(student_name: str, complaint: str) -> str:
    """
    Submit a complaint to the university.
    """
    return (
        f"Complaint Registered Successfully.\n\n"
        f"Student : {student_name}\n"
        f"Issue   : {complaint}\n"
        f"Status  : Under Review"
    )


@tool
def cgpa_gpa_calculator(current_semester_grades: list[float], previous_sgpas: list[float]) -> str:
    """
    Calculate the current semester GPA and overall CGPA.

    Args:
        current_semester_grades: Grade points (0-10) for the current semester subjects.
        previous_sgpas: SGPAs of all previously completed semesters.

    Returns:
        GPA and CGPA.
    """
    if not current_semester_grades:
        return "Please provide at least one grade point."

    gpa = round(sum(current_semester_grades) / len(current_semester_grades), 2)

    all_semesters = previous_sgpas + [gpa]
    cgpa = round(sum(all_semesters) / len(all_semesters), 2)

    return (
        f"GPA/CGPA\n"
        f"Current Semester GPA : {gpa}\n"
        f"Overall CGPA         : {cgpa}\n"
        f"Completed Semesters  : {len(all_semesters)}"
    )


@tool
def scholarship_details(cgpa: float, family_income: int, category: str) -> str:
    """
    Check scholarship eligibility based on CGPA, annual family income,
    and student category.

    Args:
        cgpa: Student's current CGPA (0-10).
        family_income: Annual family income in INR.
        category: General, OBC, SC, ST, or EWS.

    Returns:
        Scholarship eligibility details.
    """
    scholarships = []

    if cgpa >= 9.0:
        scholarships.append(
            "-Merit Scholarship\n"
            "- Eligibility: CGPA \u2265 9.0\n"
            "- Benefit: \u20b950,000 per academic year"
        )

    if cgpa >= 8.5:
        scholarships.append(
            "-Academic Excellence Scholarship\n"
            "- Eligibility: CGPA \u2265 8.5\n"
            "- Benefit: 50% Tuition Fee Waiver"
        )

    if family_income <= 300000:
        scholarships.append(
            "-Need-Based Scholarship\n"
            "- Eligibility: Family Income \u2264 \u20b93,00,000\n"
            "- Benefit: \u20b940,000 Financial Assistance"
        )

    if category.upper() in ["SC", "ST", "OBC", "EWS"]:
        scholarships.append(
            f"-{category.upper()} Category Scholarship\n"
            "- Benefit: Tuition Fee Assistance as per University Rules"
        )

    if not scholarships:
        return (
            "No scholarship is currently available based on the provided information.\n"
            "You may contact the Scholarship Office for additional schemes."
        )

    response = (
        "Scholarship Details\n\n"
        f"CGPA          : {cgpa}\n"
        f"Family Income : \u20b9{family_income:,}\n"
        f"Category      : {category.upper()}\n\n"
        "Eligible Scholarships:\n\n"
    )

    for i, scholarship in enumerate(scholarships, start=1):
        response += f"{i}. {scholarship}\n\n"

    return response


# All tools in one list, handy for registering with a LangGraph/LangChain agent
ALL_TOOLS = [
    student_profile,
    student_attendance,
    student_marks_result,
    fee_details,
    class_timetable,
    university_notices,
    faculty_information,
    course_subjects,
    library_details,
    university_policy,
    cgpa_gpa_calculator,
    scholarship_details,
    submit_leave,
    submit_complaint,
]
