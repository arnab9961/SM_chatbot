import re
import json
import torch
from transformers import GPT2Tokenizer

class ChatbotService:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.faq_questions = [faq["question"].lower() for faq in data["faq"]]
        self.faq_answers = [faq["answer"] for faq in data["faq"]]
        self.context = self._generate_context()
        
    def _generate_context(self):
        """Generate a knowledge context from the data"""
        context = []
        
        # Add company info
        company = self.data["company"]
        context.append(f"{company['name']} is {company['description']}")
        
        # Add parent company info
        parent = company["parent_company_info"]
        context.append(f"The CEO of {parent['name']} is {parent['ceo']}.")
        context.append(f"The chairperson of {parent['name']} is {parent['chairperson']}.")
        context.append(f"The owner of {parent['name']} is {parent['owner']}.")
        
        # Add management info
        management = company["management"]
        context.append(f"The General Manager of {company['name']} is {management['general_manager']}.")
        context.append(f"The GM of Sales at {company['name']} is {management['gm_sales']}.")
        
        # Add services
        services_text = f"{company['name']} offers services including: "
        services_text += ", ".join([service["name"] for service in self.data["services"]])
        context.append(services_text)
        
        # Add pricing
        for service, price in self.data["pricing"].items():
            formatted_service = service.replace('_', ' ').title()
            context.append(f"The price for {formatted_service} is {price}.")
        
        # Add tech stack
        tech_stack = ", ".join(self.data["tech_stack"])
        context.append(f"{company['name']} uses technologies including: {tech_stack}")
        
        return "\n".join(context)
    
    def _is_faq_match(self, user_input):
        """Check if the user input matches any FAQ questions"""
        user_input = user_input.lower()
        
        best_match_idx = -1
        best_match_score = 0
        
        for idx, question in enumerate(self.faq_questions):
            # Direct match
            if user_input in question or question in user_input:
                words_in_user = set(re.findall(r'\b\w+\b', user_input))
                words_in_question = set(re.findall(r'\b\w+\b', question))
                common_words = words_in_user.intersection(words_in_question)
                
                # Calculate match score based on word overlap
                score = len(common_words) / max(len(words_in_user), len(words_in_question))
                
                if score > best_match_score and score > 0.5:  # Threshold for matching
                    best_match_score = score
                    best_match_idx = idx
        
        if best_match_idx >= 0:
            return self.faq_answers[best_match_idx]
        
        return None
    
    def _extract_keyword_based_response(self, user_input):
        """Extract information based on keywords"""
        user_input = user_input.lower()
        
        # Define keyword mappings
        keyword_responses = {
            "ceo": f"The CEO of {self.data['company']['parent_company_info']['name']} is {self.data['company']['parent_company_info']['ceo']}.",
            "owner": f"The owner of {self.data['company']['parent_company_info']['name']} is {self.data['company']['parent_company_info']['owner']}.",
            "general manager": f"The General Manager of {self.data['company']['name']} is {self.data['company']['management']['general_manager']}.",
            "gm": f"The General Manager of {self.data['company']['name']} is {self.data['company']['management']['general_manager']}.",
            "sales": f"The GM of Sales at {self.data['company']['name']} is {self.data['company']['management']['gm_sales']}.",
            "service": f"{self.data['company']['name']} provides mobile app development, website development, AI solutions, CMS development, and data entry services.",
            "price": f"Website development at SM Technology starts at $2,500, and mobile app development starts at $3,500, depending on project requirements. CMS development starts at $1,000, and data entry services start at $500 per project.",
            "cost": f"Website development at SM Technology starts at $2,500, and mobile app development starts at $3,500, depending on project requirements. CMS development starts at $1,000, and data entry services start at $500 per project.",
            "technologies": f"SM Technology works with React, Next.js, Laravel, Flutter, React Native, Python, Node.js, WordPress, and Strapi.",
            "tech stack": f"SM Technology works with React, Next.js, Laravel, Flutter, React Native, Python, Node.js, WordPress, and Strapi.",
            "sister": f"The sister concerns of {self.data['company']['parent_company_info']['name']} are: Spart Tech Agency, Softvence, SM Technology, Back Bancher, Galaxy, and BdCalling Academy.",
            "parent": f"SM Technology is a sister concern of bdCalling IT, which is a global company with six sister concerns.",
            "what is": f"SM Technology is a leading IT service provider and a sister concern of bdCalling IT. It specializes in mobile app development, website development, AI solutions, CMS development, and data entry services.",
            "chairperson": f"The chairperson of {self.data['company']['parent_company_info']['name']} is {self.data['company']['parent_company_info']['chairperson']}.",
            "mobile app": f"Mobile app development at SM Technology starts at $3,500, with pricing depending on complexity and features. We create high-performance Android and iOS mobile applications using Flutter, React Native, and native technologies.",
            "website": f"Website development at SM Technology starts at $2,500, depending on project requirements. We provide custom website design and development using technologies like React, Next.js, Laravel, and WordPress."
        }
        
        # Service-specific responses
        for service in self.data["services"]:
            service_name = service["name"].lower()
            keyword_responses[service_name] = service["description"]
            
        # Technology-specific responses
        for tech in self.data["tech_stack"]:
            tech_lower = tech.lower()
            keyword_responses[tech_lower] = f"Yes, SM Technology works with {tech} for development."
        
        # Check for matches
        for keyword, response in keyword_responses.items():
            if keyword in user_input:
                return response
        
        return None
        
    def get_response(self, user_input):
        """Generate a response to the user input"""
        # Step 1: Check for FAQ matches first (highest priority)
        faq_response = self._is_faq_match(user_input)
        if faq_response:
            return faq_response
        
        # Step 2: Check for keyword-based responses
        keyword_response = self._extract_keyword_based_response(user_input)
        if keyword_response:
            return keyword_response
        
        # Step 3: Check for name-based queries (who is X?)
        if self._is_person_query(user_input):
            return self._handle_person_query(user_input)
        
        # Step 4: Check for irrelevant queries that we should reject
        irrelevant_response = self._check_irrelevant_query(user_input)
        if irrelevant_response:
            return irrelevant_response
        
        # Step 5: If we get here, try to generate a response with GPT-2
        try:
            return self._generate_gpt2_response(user_input)
        except Exception as e:
            return "I'm sorry, but I can only answer questions about SM Technology's services, management team, and company structure."

    def _is_person_query(self, user_input):
        """Check if the query is asking about a person"""
        patterns = [
            r"who is (\w+\s*\w*)",
            r"tell me about (\w+\s*\w*)",
            r"who('s| is) (\w+\s*\w*)",
            r"what do you know about (\w+\s*\w*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return True
        return False

    def _handle_person_query(self, user_input):
        """Handle queries about people"""
        # Extract the name from the query
        name = None
        patterns = [
            r"who is (\w+\s*\w*)",
            r"tell me about (\w+\s*\w*)",
            r"who('s| is) (\w+\s*\w*)",
            r"what do you know about (\w+\s*\w*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                # Get the name from the matching group
                name = match.group(1) if len(match.groups()) == 1 else match.group(2)
                break
        
        if not name:
            return "I'm not sure who you're asking about. Please provide a name."
        
        # Clean up the name
        name = name.strip().lower()
        
        # List of all known people in the dataset
        known_people = {
            "monir": "MD. Monir Hossain is the CEO and owner of bdCalling IT, the parent company of SM Technology.",
            "hossain": "MD. Monir Hossain is the CEO and owner of bdCalling IT, the parent company of SM Technology.",
            "monir hossain": "MD. Monir Hossain is the CEO and owner of bdCalling IT, the parent company of SM Technology.",
            "md. monir hossain": "MD. Monir Hossain is the CEO and owner of bdCalling IT, the parent company of SM Technology.",
            "md monir hossain": "MD. Monir Hossain is the CEO and owner of bdCalling IT, the parent company of SM Technology.",
            
            "sabina": "MST. Sabina Akter is the Chairperson of bdCalling IT, the parent company of SM Technology.",
            "akter": "MST. Sabina Akter is the Chairperson of bdCalling IT, the parent company of SM Technology.",
            "sabina akter": "MST. Sabina Akter is the Chairperson of bdCalling IT, the parent company of SM Technology.",
            "mst. sabina akter": "MST. Sabina Akter is the Chairperson of bdCalling IT, the parent company of SM Technology.",
            "mst sabina akter": "MST. Sabina Akter is the Chairperson of bdCalling IT, the parent company of SM Technology.",
            
            "shamim": "MD. Shamim Miah is the General Manager of SM Technology.",
            "miah": "MD. Shamim Miah is the General Manager of SM Technology.",
            "shamim miah": "MD. Shamim Miah is the General Manager of SM Technology.",
            "md. shamim miah": "MD. Shamim Miah is the General Manager of SM Technology.",
            "md shamim miah": "MD. Shamim Miah is the General Manager of SM Technology.",
            
            "jabed": "MD. Jabed is the GM of Sales at SM Technology.",
            "md. jabed": "MD. Jabed is the GM of Sales at SM Technology.",
            "md jabed": "MD. Jabed is the GM of Sales at SM Technology."
        }
        
        # Company names
        known_companies = {
            "sm technology": "SM Technology is a leading IT service provider and a sister concern of bdCalling IT. It specializes in mobile app development, website development, AI solutions, CMS development, and data entry services.",
            "sm": "SM Technology is a leading IT service provider and a sister concern of bdCalling IT. It specializes in mobile app development, website development, AI solutions, CMS development, and data entry services.",
            "bdcalling": "bdCalling IT is a global company with six sister concerns, including SM Technology. The CEO is MD. Monir Hossain and the Chairperson is MST. Sabina Akter.",
            "bdcalling it": "bdCalling IT is a global company with six sister concerns, including SM Technology. The CEO is MD. Monir Hossain and the Chairperson is MST. Sabina Akter.",
            "spart tech": "Spart Tech Agency is a sister concern of bdCalling IT.",
            "spart tech agency": "Spart Tech Agency is a sister concern of bdCalling IT.",
            "softvence": "Softvence is a sister concern of bdCalling IT.",
            "back bancher": "Back Bancher is a sister concern of bdCalling IT.",
            "galaxy": "Galaxy is a sister concern of bdCalling IT.",
            "bdcalling academy": "BdCalling Academy is a sister concern of bdCalling IT."
        }
        
        # Check if the name is a known person
        if name in known_people:
            return known_people[name]
        
        # Check if the name is a known company
        if name in known_companies:
            return known_companies[name]
        
        # If we don't have information about this person or company
        return f"I don't have any information about {name}. I can only provide information about the management team of SM Technology and bdCalling IT."

    def _check_irrelevant_query(self, user_input):
        """Check if the query is irrelevant to our dataset"""
        user_input_lower = user_input.lower()
        
        # List of irrelevant topics
        irrelevant_topics = [
            "weather", "stock", "sport", "game", "movie", "music", "food", "restaurant",
            "hotel", "flight", "train", "bus", "car", "bike", "book", "novel",
            "news", "politics", "election", "president", "prime minister", "police", "crime",
            "accident", "health", "doctor", "hospital", "medicine", "covid", "virus",
            "vaccine", "investment", "bitcoin", "crypto", "blockchain"
        ]
        
        # Check if query contains irrelevant topics
        for topic in irrelevant_topics:
            if topic in user_input_lower:
                return "I'm sorry, I can only answer questions about SM Technology, its services, management team, and company structure."
        
        # Check for personal questions
        personal_patterns = [
            r"(how are|how're|how do) you",
            r"what('s| is) your name",
            r"who are you",
            r"tell me about yourself",
            r"where are you",
            r"how old are you",
            r"what do you (like|love|enjoy|prefer)",
            r"your (favorite|favourite)"
        ]
        
        for pattern in personal_patterns:
            if re.search(pattern, user_input_lower):
                return "I'm an AI chatbot designed to provide information about SM Technology, its services, and management team."
        
        # No irrelevant topics found
        return None

    def _generate_gpt2_response(self, user_input):
        """Generate a response using GPT-2"""
        # Prepare prompt with context
        prompt = f"Information about SM Technology:\n{self.context}\n\nQuestion: {user_input}\nAnswer:"
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate with GPT-2
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                top_k=30,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract the answer part
        if "Answer:" in generated_text:
            response = generated_text.split("Answer:")[1].strip()
            # Clean up and format response
            if "\n" in response:
                response = response.split("\n")[0]
        else:
            return "I'm sorry, I can only answer questions about SM Technology, its services, management team, and company structure."
        
        # Filter out nonsensical or fabricated responses
        red_flags = [
            "I am", "I have", "My name", "I'm a", "I work", "I don't",
            "BDD", "blog post", "years", "fan",
            "The CEO is the CEO", "The Chairman is the Chairman",
            "MD.", "Mr.", "Dr."  # Be cautious with titles not in our dataset
        ]
        
        for flag in red_flags:
            if flag in response and not any(flag in person for person in [
                "MD. Monir Hossain", "MD. Shamim Miah", "MD. Jabed", "MST. Sabina Akter"
            ]):
                return "I'm sorry, I can only answer questions about SM Technology, its services, management team, and company structure."
        
        # Check for reasonable length
        if len(response) > 200 or len(response) < 10:
            return "I'm sorry, I can only answer questions about SM Technology, its services, management team, and company structure."
        
        return response
