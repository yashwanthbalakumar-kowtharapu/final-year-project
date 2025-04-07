import google.generativeai as genai

def generate_review(job_role,company, questions, answers, emotion, suspicious_count):
    try:
        # Construct the improved prompt
        prompt = f"""
        You are an experienced interviewer for a {job_role} position at {company} company.
        Below are the interview questions and answers provided by the candidate.

        For each question:
        Question 1:[question text]
        Answer : [candidate answer]
        Feedback :[] detail feed back on answer and suggestions for improvement]
        [repeat for each question]
        - Evaluate the answer.
        - Provide feedback on technical accuracy, completeness, and clarity.
        - Suggest improvements if necessary.

        Also, include:
        - A final summary of the candidate's overall performance.
        - A recommendation on whether they should be considered for the role.

        Interview Data:
        Questions: {questions}
        company:{company}
        Answers: {answers}
        Emotion detected: {emotion}
        Suspicious behavior count: {suspicious_count}
        """

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)

        # Extract response text safely
        review_text = response.text if response else "Review generation failed."

        return review_text
    except Exception as e:
        return f"Error: {e}"
