from dotenv import load_dotenv
import os
import google.generativeai as genai
from typing import Dict, List, Union
import json
import re
import logging
import gunicorn
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import requests
from datetime import datetime

class ResponseFormatter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def preserve_formatting(self, text: str) -> str:
        """Preserve markdown formatting while adding structure"""
        # Preserve existing ** formatting
        formatted_text = text.replace('**', '||BOLD||')  # Temporarily replace ** with marker
        
        # Add formatting to headers if they don't exist
        lines = formatted_text.split('\n')
        formatted_lines = []
        for line in lines:
            # Add bold to headers if they don't already have formatting
            if ':' in line and not '||BOLD||' in line:
                header, content = line.split(':', 1)
                formatted_lines.append(f"**{header}:**{content}")
            else:
                formatted_lines.append(line)
        
        # Restore ** formatting
        formatted_text = '\n'.join(formatted_lines)
        formatted_text = formatted_text.replace('||BOLD||', '**')
        return formatted_text

    def format_academic_response(self, text: str) -> str:
        """Format academic advice responses with bullet points and sections"""
        try:
            sections = text.split("\n\n")
            formatted_sections = []
            
            # Add header to first section
            if sections[0]:
                formatted_sections.append(f"**📚 Overview:**\n{sections[0]}")
            
            # Format remaining sections
            for section in sections[1:]:
                if len(section.strip()) > 0:
                    # Preserve existing formatting while adding structure
                    formatted_section = self.preserve_formatting(section)
                    if not formatted_section.strip().startswith('•'):
                        formatted_section = "• " + formatted_section.replace("\n", "\n• ")
                    formatted_sections.append(formatted_section)
            
            return "\n\n".join(formatted_sections)
        except Exception as e:
            self.logger.error(f"Error formatting academic response: {str(e)}")
            return text

    def format_moral_response(self, text: str) -> str:
        """Format moral support responses with empathetic structure"""
        try:
            paragraphs = text.split("\n\n")
            formatted_parts = []
            
            if len(paragraphs) >= 1:
                formatted_parts.append(f"**💭 Response:**\n{self.preserve_formatting(paragraphs[0])}")
            
            if len(paragraphs) >= 2:
                formatted_parts.append(f"**💡 Suggestion:**\n{self.preserve_formatting(paragraphs[1])}")
            
            for para in paragraphs[2:]:
                formatted_para = self.preserve_formatting(para)
                if "step" in para.lower() or "tip" in para.lower():
                    formatted_parts.append(f"**✨ {formatted_para}**")
                else:
                    formatted_parts.append(f"• {formatted_para}")
            
            return "\n\n".join(formatted_parts)
        except Exception as e:
            self.logger.error(f"Error formatting moral response: {str(e)}")
            return text

    def format_university_response(self, text: str) -> str:
        """Format university life responses with practical sections"""
        try:
            sections = text.split("\n\n")
            formatted_sections = []
            
            for i, section in enumerate(sections):
                formatted_section = self.preserve_formatting(section)
                if i == 0:
                    formatted_sections.append(f"**🎓 Overview:**\n{formatted_section}")
                elif "resource" in section.lower():
                    formatted_sections.append(f"**📚 Resources:**\n{formatted_section}")
                elif any(word in section.lower() for word in ["tip", "advice", "suggestion"]):
                    formatted_sections.append(f"**💡 Helpful Tips:**\n{formatted_section}")
                else:
                    formatted_sections.append(f"• {formatted_section}")
            
            return "\n\n".join(formatted_sections)
        except Exception as e:
            self.logger.error(f"Error formatting university response: {str(e)}")
            return text
class UniversityRecommender:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.logger = logging.getLogger(__name__)

    def _get_university_image(self, university_name: str, country: str) -> str:
        """Generate image URL for a university"""
        formattedname = ""
        formattedcountry = ""
        for i in university_name:
            if i == ' ' or i == '+':
                formattedname += '+'
            else:
                formattedname += i
            
        for i in country:
            if i == ' ' or i == '+':
                formattedcountry += '+'
            else:
                formattedcountry += i
        search_query = f"{formattedname}+4k+building+picture"
        imageapikey = os.getenv('IMAGE_API_KEY')
        print(search_query)
        coolquery = f"https://www.googleapis.com/customsearch/v1?key=AIzaSyDptyzxGJg-aR5IldozvISzjNgF2_TISJo&cx=e1cac863f07bf4f8b&q={search_query}&searchType=image"
        imageresponse = requests.get(coolquery).json()
        self.logger.warning(imageresponse.get('items')[0].get('link'))
        return imageresponse.get('items')[0].get('link') #"https://www.pokemon.com/static-assets/content-assets/cms2/img/pokedex/full/813.png"
    # def _get_university_image(self, university_name: str, country: str) -> str:   
    #     return "https://www.pokemon.com/static-assets/content-assets/cms2/img/pokedex/full/813.png"
    def _generate_university_label(self, university_name: str, faculty_strengths: str) -> List[Dict[str, str]]:
        """Generate single best badge for university achievement"""
        try:
            # Generate focused badge using Gemini
            prompt = f"""
            For {university_name}, with strengths in {faculty_strengths},
            generate the SINGLE most prestigious ranking or achievement in this JSON format:
            [
                {{
                    "type": "ranking",
                    "text": "badge text",
                    "details": "specific details",
                    "field": "field name"
                }}
            ]

            Rules for badge generation:
            1. MUST only generate the SINGLE most impressive ranking
            2. Prioritize order:
            - World ranking if in top 50 globally (format as "#X Globally")
            - Program ranking if #1-3 in world (e.g., "#1 in CS")
            - Regional leadership if #1 in region (e.g., "#1 in Europe")
            - Specific achievement if no top rankings
            3. Text must be 2-4 words maximum
            4. Must be extremely specific
            5. Include real ranking where possible
            
            Examples:
            - {{"type": "world_rank", "text": "#15 Globally", "details": "QS World Rankings 2024", "field": "Overall"}}
            - {{"type": "faculty_rank", "text": "#1 in AI", "details": "World's leading AI research center", "field": "Computer Science"}}
            - {{"type": "region_rank", "text": "#1 in Asia", "details": "Leading Asian institution", "field": "Regional Standing"}}
            """

            response = self.model.generate_content(prompt)
            badges = json.loads(response.text)
            
            # Take only the first (best) badge
            if len(badges) > 0:
                badge = badges[0]
            else:
                return [{
                    "text": "Featured Program",
                    "details": "Notable academic institution",
                    "field": "Overall",
                    "color_class": "bg-purple-600",
                    "prefix": "✨"
                }]

            # Updated badge styling map
            type_styling = {
                "world_rank": {
                    "colors": {
                        "1": "bg-yellow-500",     # Gold for #1
                        "2": "bg-gray-400",       # Silver for #2
                        "3": "bg-orange-500",     # Bronze for #3
                        "top10": "bg-blue-500",   # Blue for top 10
                        "top50": "bg-indigo-500", # Indigo for top 50
                    },
                    "prefix": "🌍"
                },
                "faculty_rank": {
                    "colors": {
                        "1": "bg-yellow-500",
                        "2": "bg-gray-400",
                        "3": "bg-orange-500",
                        "default": "bg-blue-500"
                    },
                    "prefix": "🎯"
                },
                "region_rank": {
                    "colors": {
                        "1": "bg-green-500",
                        "default": "bg-green-600"
                    },
                    "prefix": "🏆"
                }
            }

            # Extract ranking number if present
            ranking_match = re.search(r'#(\d+)', badge["text"])
            ranking_number = ranking_match.group(1) if ranking_match else None

            badge_type = badge.get("type", "world_rank")
            styling = type_styling.get(badge_type, type_styling["world_rank"])

            # Determine color class based on ranking and type
            if ranking_number:
                if badge_type == "world_rank":
                    if ranking_number == "1":
                        color_class = styling["colors"]["1"]
                    elif ranking_number == "2":
                        color_class = styling["colors"]["2"]
                    elif ranking_number == "3":
                        color_class = styling["colors"]["3"]
                    elif int(ranking_number) <= 10:
                        color_class = styling["colors"]["top10"]
                    else:
                        color_class = styling["colors"]["top50"]
                else:
                    color_class = styling["colors"].get(ranking_number, styling["colors"].get("default", "bg-blue-500"))
            else:
                color_class = styling["colors"].get("default", "bg-blue-500")

            return [{
                "text": badge["text"],
                "details": badge["details"],
                "field": badge["field"],
                "color_class": color_class,
                "prefix": styling["prefix"]
            }]

        except Exception as e:
            self.logger.error(f"Error generating university label: {str(e)}")
            return [{
                "text": "Featured Program",
                "details": "Notable academic institution",
                "field": "Overall",
                "color_class": "bg-purple-600",
                "prefix": "✨"
            }]
    def get_university_details(self, university_name: str, country: str) -> Dict:
        """Get detailed information about a university"""
        try:
            prompt = f"""
            Please provide detailed information about {university_name} in {country}.
            Return the response in this exact JSON format:
            {{
                "description": "Brief overview",
                "notablePrograms": "List of programs",
                "campusLife": "Description of campus life",
                "admissionRequirements": "Admission details",
                "researchOpportunities": "Research facilities and options",
                "ranking": "Current rankings"
            }}
            """

            response = self.model.generate_content(prompt)
            if not response.text:
                return {"error": "No response received from AI model"}

            # Clean and sanitize the response text
            cleaned_text = response.text.strip()
            
            # Remove code block markers if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            # Replace problematic characters
            cleaned_text = (
                cleaned_text
                .replace('\n', ' ')
                .replace('\r', '')
                .replace('\t', ' ')
            )
            
            # Clean up multiple spaces
            cleaned_text = ' '.join(cleaned_text.split())
            
            try:
                # First attempt: direct JSON parse
                details = json.loads(cleaned_text)
            except json.JSONDecodeError:
                try:
                    # Second attempt: Use regex to extract JSON object
                    json_match = re.search(r'{.*}', cleaned_text, re.DOTALL)
                    if json_match:
                        details = json.loads(json_match.group())
                    else:
                        raise ValueError("No valid JSON object found")
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.error(f"JSON parsing error: {str(e)}\nCleaned text: {cleaned_text}")
                    return {
                        "error": "Failed to parse university details",
                        "description": "Error retrieving university information",
                        "notablePrograms": "Information unavailable",
                        "campusLife": "Information unavailable",
                        "admissionRequirements": "Information unavailable",
                        "researchOpportunities": "Information unavailable",
                        "ranking": "Information unavailable"
                    }

            # Validate required fields
            required_fields = [
                "description", "notablePrograms", "campusLife",
                "admissionRequirements", "researchOpportunities", "ranking"
            ]
            
            # Ensure all required fields exist
            for field in required_fields:
                if field not in details:
                    details[field] = "Information unavailable"
                    
                    # Add image URL and label
           # Validate required fields
            required_fields = [
                "description", "notablePrograms", "campusLife",
                "admissionRequirements", "researchOpportunities", "ranking"
            ]
            
            # Ensure all required fields exist
            for field in required_fields:
                if field not in details:
                    details[field] = "Information unavailable"
            
            # Add image URL
            details['imageUrl'] = self._get_university_image(university_name, country)
            
            # Generate and add label
            details['label'] = self._generate_university_label(university_name, details.get('notablePrograms', ''))
            
            return details

        except Exception as e:
            self.logger.error(f"Error fetching university details: {str(e)}")
            return {
                "error": f"Error fetching university details: {str(e)}",
                "description": "Error retrieving university information",
                "notablePrograms": "Information unavailable",
                "admissionRequirements": "Information unavailable",
                "researchOpportunities": "Information unavailable",
                "ranking": "Information unavailable"
            }

    def recommend(self, country: str, faculty: str = None, gpa: str = None,
                 budget: str = None, sat: str = None, extra: str = None) -> Union[List[Dict], Dict[str, str]]:
        """Generate university recommendations"""
        if not country:
            return {"error": "Country is required"}

        try:
            prompt = f"""
            Please recommend exactly 6 universities in {country} that match these criteria.
            Return the response in this exact JSON array format:
            [
                {{
                    "universityName": "University Name",
                    "location": "City, {country}",
                    "tuition": "Amount per year",
                    "acceptanceRate": "XX%",
                    "GPA": "X.XX",
                    "facultyStrengths": "List of strong programs"
                }}
            ]
            """

            if faculty:
                prompt += f"\nField of study: {faculty}"
            if gpa:
                prompt += f"\nStudent GPA: {gpa}"
            if sat:
                prompt += f"\nSAT score: {sat}"
            if budget:
                prompt += f"\nBudget: {budget}"
            if extra:
                prompt += f"\nAdditional requirements: {extra}"

            prompt += "\nPlease ensure the response is ONLY the JSON array with exactly 6 universities."

            response = self.model.generate_content(prompt)
            if not response or not response.text:
                return {"error": "No response received from AI model"}

            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            try:
                recommendations = json.loads(cleaned_text)
                
                # Validate response structure
                if not isinstance(recommendations, list):
                    return {"error": "Invalid response format - expected a list"}
                    
                if len(recommendations) != 6:
                    return {"error": f"Expected 6 universities, got {len(recommendations)}"}

                # Process each university
                for uni in recommendations:
                    # Validate required fields
                    required_fields = ["universityName", "location", "tuition", "acceptanceRate", "GPA", "facultyStrengths"]
                    if not all(key in uni for key in required_fields):
                        return {"error": "Invalid university data structure - missing required fields"}

                    # Process faculty strengths
                    if isinstance(uni.get('facultyStrengths'), list):
                        uni['facultyStrengths'] = ', '.join(str(f).strip("'[]") for f in uni['facultyStrengths'])
                    elif isinstance(uni.get('facultyStrengths'), str):
                        uni['facultyStrengths'] = uni['facultyStrengths'].strip("'[]")

                    # Add image URL and label
                    try:
                        university_name = uni.get('universityName', '')
                        uni['imageUrl'] = self._get_university_image(university_name, country)
                        uni['label'] = self._generate_university_label(university_name, uni['facultyStrengths'])
                    except Exception as e:
                        self.logger.error(f"Error getting image/label for {university_name}: {str(e)}")
                        uni['imageUrl'] = "default_university_image_url.jpg"  # Fallback image URL
                        uni['label'] = {
                            "text": "Featured University",
                            "color_class": "bg-[#9333EA]",
                            "prefix": "✨"
                        }

                return recommendations

            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {str(e)}\nText attempted to parse: {cleaned_text}")
                return {"error": "Failed to parse AI response as JSON"}

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {"error": f"Error generating recommendations: {str(e)}"}

    def get_career_guidance(self, question_count: int, previous_prompt: str = "") -> tuple:
        """Generate AI-driven career guidance questions and process responses"""
        try:
            if question_count == 6:
                analysis_prompt = f"""Based on the following conversation, provide a detailed career analysis in this JSON format:
                {{
                    "profile_traits": [
                        {{"name": "trait name", "description": "trait description"}},
                        // 3-4 key traits based on the conversation
                    ],
                    "specializations": [
                        {{
                            "title": "specialization name",
                            "description": "detailed description",
                            "skills": ["skill1", "skill2", "skill3"],
                            "recommended_courses": ["course1", "course2"]
                        }},
                        // exactly 3 specializations ordered by best match
                    ],
                    "summary": "brief summary of career direction"
                }}

                Previous conversation: {previous_prompt}
                """
                
                response = self.model.generate_content(analysis_prompt)
                try:
                    cleaned_text = response.text.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]
                        
                    analysis = json.loads(cleaned_text)
                    return analysis, ""
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse career analysis JSON: {e}")
                    return {"error": "Failed to analyze career guidance"}, ""

            if question_count == 1:
                prompt = """You are a career guidance counselor. Create a thought-provoking first question about the student's interests, passions, and values,but make the questions simple.
                Format your response exactly as: Question
                Make the question specific and engaging."""
            else:
                prompt = f"""Based on this conversation history:
                {previous_prompt}
                
                Generate the next career guidance question that builds upon previous answers.
                Make the question more specific and focused on career direction.
                Consider previous responses to make questions more relevant.
                
                Format your response exactly as: Question"""

            response = self.model.generate_content(prompt)
            if not response.text:
                return "Error generating question", ""
                
            parts = response.text.split('(sep)')
            return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
                
        except Exception as e:
            self.logger.error(f"Error in career guidance: {str(e)}")
            return "Error generating question", ""
# Initialize Flask application
app = Flask(__name__)
app.secret_key = "dev_secret_key_123"
logging.basicConfig(level=logging.INFO)

# Initialize recommender and formatter
recommender = UniversityRecommender()
response_formatter = ResponseFormatter()

@app.route('/')
def index():
    return render_template("mainpage.html")

@app.route('/advisor')
def advisor():
    return render_template("advisor.html")
@app.route('/chat', methods=['POST'])
def chat():
    message = request.form.get('message')
    university_name = request.form.get('university_name')
    print(f"Received message: {message}")  # Debug print
    print(f"For university: {university_name}")  # Debug print
    
    if not message or not university_name:
        return jsonify({"error": "Message and university name are required"}), 400
    
    try:
        prompt = f"""Act as an AI advisor with specific knowledge about {university_name}. 
        Answer this question: {message}
        
        Provide specific details about {university_name} relevant to the question.
        Include information about programs, campus life, requirements, or other relevant aspects.
        Use clear sections with markdown **bold** for headers."""
        
        print(f"Sending prompt to model: {prompt}")  # Debug print
        response = recommender.model.generate_content(prompt)
        print(f"Model response: {response.text}")  # Debug print
        
        if not response.text:
            return jsonify({"error": "Failed to generate response"}), 500
            
        formatted_response = response_formatter.format_university_response(response.text)
        print(f"Formatted response: {formatted_response}")  # Debug print
        
        return jsonify({
            "response": formatted_response,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Debug print
        return jsonify({
            "error": "Failed to generate response",
            "details": str(e)
        }), 500
@app.route('/university_details', methods=['POST'])
def university_details():
    university_name = request.form.get('university_name')
    country = request.form.get('country')
    
    if not university_name or not country:
        return jsonify({"error": "University name and country are required"}), 400
        
    details = recommender.get_university_details(university_name, country)
    return jsonify(details)

@app.route('/unichooser', methods=['GET', 'POST'])
def unichooser():
    if request.method == 'POST':
        form_data = {
            'country': request.form.get('country', '').strip(),
            'faculty': request.form.get('faculty', '').strip(),
            'gpa': request.form.get('gpa', '').strip(),
            'budget': request.form.get('budget', '').strip(),
            'sat': request.form.get('sat', '').strip(),
            'extra': request.form.get('extra', '').strip()
        }
        
        form_data = {k: v for k, v in form_data.items() if v}
        app.logger.info(f"Processing recommendation request: {form_data}")
        
        if not form_data.get('country'):
            return render_template("index.html", error="Country is required")
        
        try:
            recommendations = recommender.recommend(**form_data)
            app.logger.info(f"Generated recommendations: {recommendations}")
            
            if isinstance(recommendations, dict) and 'error' in recommendations:
                return render_template("index.html", error=recommendations['error'])
            
            return render_template(
                "index.html", 
                recommendations=recommendations,
                country=form_data['country']
            )
            
        except Exception as e:
            app.logger.error(f"Error processing recommendations: {str(e)}")
            return render_template(
                "index.html",
                error=f"An error occurred while processing your request: {str(e)}"
            )
    
    return render_template("index.html")

@app.route('/loadquiz', methods=['GET', 'POST'])
def career_quiz():
    if request.method == 'GET':
        # Start new quiz
        question, description = recommender.get_career_guidance(1)
        session['quiz_started'] = datetime.now().isoformat()
        
        return render_template(
            'quiz.html',
            question=question,
            description=description,
            count=1,
            newprompt=' You asked: ' + question
        )
    
    count = int(request.form.get('hid', 1))
    answer = request.form.get('answer', '')
    previous_prompt = request.form.get('hidprompt', '')
    new_prompt = previous_prompt + ' I answered: ' + answer

    try:
        if count >= 6:
            # Get final analysis
            analysis, _ = recommender.get_career_guidance(count, new_prompt)
            
            if isinstance(analysis, dict):
                if 'error' in analysis:
                    return render_template(
                        'specialty.html',
                        error=True
                    )
                
                # Store analysis in session for later use
                session['career_analysis'] = analysis
                
                return render_template(
                    'specialty.html',
                    error=False,
                    profile_traits=analysis.get('profile_traits', []),
                    specializations=analysis.get('specializations', []),
                    summary=analysis.get('summary', '')
                )

        # Generate next question
        question, description = recommender.get_career_guidance(count, new_prompt)
        
        if isinstance(question, dict) and 'error' in question:
            return render_template(
                'quiz.html',
                question="An error occurred. Please try again.",
                description='',
                count=count,
                newprompt=new_prompt
            )
            
        return render_template(
            'quiz.html',
            question=question,
            description=description,
            count=count + 1,
            newprompt=str(new_prompt) + '; You asked: ' + str(question)
        )
        
    except Exception as e:
        app.logger.error(f"Error in career quiz: {str(e)}")
        return render_template(
            'specialty.html',
            error=True
        )
    
@app.route('/apply-recommendations', methods=['POST'])
def apply_recommendations():
    """Apply career recommendations to university search"""
    if 'career_analysis' not in session:
        return jsonify({"error": "No career analysis found"}), 400
        
    analysis = session['career_analysis']
    
    # Get primary specialization if available
    if analysis.get('specializations') and len(analysis['specializations']) > 0:
        primary_spec = analysis['specializations'][0]['title']
    else:
        primary_spec = ""
    
    # Redirect to university search with pre-filled parameters
    return redirect(url_for('unichooser', faculty=primary_spec))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Configure API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
    
    # Run the application
    app.run(debug=True, port=3000)