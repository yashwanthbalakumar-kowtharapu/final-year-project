import google.generativeai as genai

def generate_questions(job_role, experience_lvl, company):
    try:
        # Construct the prompt for question generation
        prompt = f"""Generaten 10 to 20 random technical interview questions for a {job_role} position with {experience_lvl} experience level for the company {company}.
        The questions should:
        1. Be relevant to the role
        2. Match the experience level
        3. Cover different technical aspects
        4. Be clear and specific
        5. Be in the style of a technical interview
        6.starts with very easy humanresource round and technical questions
        7.most asked question acoording to {job_role} in the company {company}
        make shure that the questions are unique  
     important note: Return the questions as a list.no need of any other explantions just give questions only"""

        # Get model from global context
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Process the response into a list of questions
        questions = response.text.strip().split('\n')
        # Clean up the questions (remove numbering if present)
        questions = [q.strip().lstrip('0123456789.)-') for q in questions if q.strip()]
        
        return questions  # Return exactly 5 questions
        
    except Exception as e:
        return f"Error generating questions: {str(e)}" 