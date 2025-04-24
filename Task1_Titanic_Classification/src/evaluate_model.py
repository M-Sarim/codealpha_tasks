import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import numpy as np

def generate_pdf(report, confusion_matrix_str, accuracy, filename='evaluation_report.pdf'):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Model Evaluation Report")

    # Accuracy
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    c.setFont("Helvetica", 10)
    y_position = 680
    c.drawString(100, y_position, "Classification Report:")
    y_position -= 20
    for line in report.splitlines():
        c.drawString(100, y_position, line)
        y_position -= 15

    # Confusion Matrix
    c.drawString(100, y_position, "Confusion Matrix:")
    y_position -= 20
    for line in confusion_matrix_str.splitlines():
        c.drawString(100, y_position, line)
        y_position -= 15

    c.save()

def main():
    # Read the validation data
    val = pd.read_csv('data/processed/val_processed.csv')
    X_val = val.drop('Survived', axis=1)
    y_val = val['Survived']

    # Load the trained model
    model = load('models/titanic_model.pkl')

    # Make predictions
    preds = model.predict(X_val)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, preds)
    cm = confusion_matrix(y_val, preds)
    report = classification_report(y_val, preds)

    # Convert confusion matrix to string
    cm_str = '\n'.join(['\t'.join(map(str, row)) for row in cm])

    # Generate PDF
    generate_pdf(report, cm_str, accuracy)

    print("Evaluation report saved as PDF.")

if __name__ == '__main__':
    main()
