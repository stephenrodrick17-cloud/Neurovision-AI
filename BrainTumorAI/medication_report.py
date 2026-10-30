import os
from datetime import datetime
from fpdf import FPDF

def suggest_treatment(_scenario):
    _ = _scenario
    return {
        "description": "Treatment examples are not provided by this research prototype.",
        "treatments": ["No treatment advice is generated."],
        "followup": "No follow-up guidance is generated.",
    }

class MedicalReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(30, 60, 120)  # Medical Blue
        self.cell(0, 10, 'NEUROVISION AI - DEMONSTRATION SUMMARY', 0, 1, 'C')
        self.ln(5)
        self.set_draw_color(30, 60, 120)
        self.line(10, 25, 200, 25)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | Prototype output - not a clinical report', 0, 0, 'C')

def create_pdf_report(patient_name, scan_id, classification, confidence, location, size_cm, notes):
    pdf = MedicalReportPDF()
    pdf.add_page()
    
    # Patient Info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'DEMONSTRATION RECORD CONTEXT', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(100, 7, f'Demo record name: {patient_name}', 0, 0)
    pdf.cell(0, 7, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
    pdf.cell(100, 7, f'Demo identifier: {scan_id}', 0, 1)
    pdf.ln(5)
    
    # Findings
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'BINARY IMAGE-CLASSIFIER OUTPUT', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    status_color = (30, 60, 120)
    pdf.set_text_color(*status_color)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 7, f'Folder-label model output: {classification.upper()}', 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 10)
    
    pdf.cell(100, 7, f'Top-class softmax score (uncalibrated): {confidence:.2f}%', 0, 1)
    if classification != "no_tumor":
        pdf.cell(100, 7, f'Anatomical location: Not assessed ({location})', 0, 1)
        pdf.cell(100, 7, f'Lesion size/volume: Not measured ({size_cm})', 0, 1)
    pdf.ln(5)
    
    # Treatment
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'INTERPRETATION LIMITS', 0, 1, 'L')
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(
        0,
        7,
        'This prototype classifies a single 2D image using folder-derived labels. It does not identify '
        'tumor subtype, measure lesions, localize anatomy, recommend treatment, or provide clinical decision support.',
    )
    pdf.ln(5)
    
    # Notes
    if notes:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'ADDITIONAL NOTES', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 7, notes)
        pdf.ln(5)
    
    # Disclaimer
    pdf.set_y(-40)
    pdf.set_font('Arial', 'I', 7)
    pdf.set_text_color(100, 100, 100)
    disclaimer = (
        "RESEARCH / EDUCATION DEMONSTRATION ONLY. Not a diagnosis, clinical report, anatomical localization, "
        "treatment recommendation, or validated medical-device output. Do not use for patient-care decisions."
    )
    pdf.multi_cell(0, 4, disclaimer)
    
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"Demo_Summary_{scan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(filename)
    return filename

def generate_report_text(patient_name, scan_id, classification, size_cm=None, location=None):
    return f"""
    NEUROVISION AI - DEMONSTRATION SUMMARY
    ----------------------------
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Demo record name: {patient_name}
    Demo identifier: {scan_id}
    
    BINARY IMAGE-CLASSIFIER OUTPUT:
    -------------------
    Folder-label class output: {classification}
    Top-class score: Not reported as calibrated probability
    Lesion size/volume: Not measured ({size_cm if size_cm else 'N/A'})
    Anatomical location: Not assessed ({location if location else 'N/A'})
    
    This software does not identify tumor subtype, measure lesions, localize anatomy, or recommend treatment.
    RESEARCH / EDUCATION DEMONSTRATION ONLY — NOT FOR PATIENT-CARE DECISIONS.
    """

if __name__ == "__main__":
    pass
