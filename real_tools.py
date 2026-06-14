import requests
import wikipedia
import json
import os

# Configure your API KEY
# Register at https://serper.dev/ to get 2500 free calls
SERPER_API_KEY = "" 

class RealTools:
    @staticmethod
    def search_google(query):
        """
        Perform Google search using Serper API
        """
        print(f"   [?? Search] Googling: {query}...")
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": 3  # Only get top 3 results to save tokens
        })
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            data = response.json()
            
            # Extract snippets
            snippets = []
            if 'organic' in data:
                for result in data['organic']:
                    if 'snippet' in result:
                        snippets.append(result['snippet'])
            
            if not snippets:
                return "No relevant search results found."
            
            # Merge top 3 snippets
            return " ".join(snippets)
            
        except Exception as e:
            print(f"   [Warning] Serper Search failed: {e}")
            return "Search failed."

    @staticmethod
    def search_wikipedia(keyword):
        """
        Search Wikipedia summary
        """
        print(f"   [?? Wiki] Searching: {keyword}...")
        try:
            # Search for the most relevant page
            search_results = wikipedia.search(keyword)
            if not search_results:
                return ""
            
            # Get summary of the first result
            page = wikipedia.page(search_results[0], auto_suggest=False)
            return page.summary[:500] + "..." # Limit length to prevent context overflow
            
        except wikipedia.DisambiguationError as e:
            # If there's ambiguity, select the first option
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return page.summary[:500] + "..."
            except:
                return ""
        except Exception as e:
            print(f"   [Warning] Wiki failed: {e}")
            return ""

    @staticmethod
    def get_knowledge(coarse_label, question):
        """
        Intelligent combined search: First search Wiki, if no results from Wiki, then search Google
        """
        # 1. Prioritize Wiki search (purer knowledge source)
        wiki_info = RealTools.search_wikipedia(coarse_label)
        
        # 2. If Wiki content is too limited or the question is specific, search Google for the question
        google_info = ""
        # For counting or action questions, no need for extensive knowledge - mainly rely on image analysis
        if "how many" not in question.lower() and "doing" not in question.lower():
             # Construct search term: e.g., "Black-footed Albatross features"
             google_info = RealTools.search_google(f"{coarse_label} visual features identification")
        
        return f"Wikipedia Info: {wiki_info}\nGoogle Search Info: {google_info}"