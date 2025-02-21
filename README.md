# Concept Similarity Tool

This tool is designed to provide interesting insights into the field of machine learning by connecting concepts that are closer to our daily lives than we might think. 
It allows us to visualize relationships between concepts using a graph-based approach, and the tool provides a way to bring these relationships closer through vector space models.

## Key Design Decisions

The application primarily works by integrating SPARQL queries and building hierarchies of categories using these queries. 
It then calculates concept similarity by comparing the relationships between these categories, enabling us to understand how closely related certain terms are in our daily life. 
This is done by treating these relationships as if they are part of a neural network.

### Integration of SPARQL Queries:
- SPARQL queries are implemented to fetch current and parent categories for a given resource or URI from the DBpedia database.

### Hierarchy Construction:
- An algorithm is developed to perform a breadth-first search (BFS) or other search algorithms to generate the category hierarchy up to a certain depth.
- The algorithm also calculates the depth of the categories and merges results for multiple categories.

## Key Endpoints

### /api/v1/categories/resource
This asynchronous function is responsible for extracting category hierarchies for a given resource and recursively fetching categories for each result found in the graph. 
It returns all parent and child categories of the resource, its depth, and the number of nodes. The BFS algorithm is used to arrange the nodes (categories) in layers within the graph around a single resource. 
The distance between the starting root node and the final category is calculated based on depth. A sample input could be: `<Barack_Obama>`. Output -> /screens/Screenshot_3.png

### /api/v1/concepts_cohesion/resource1/resource2/query
This endpoint analyzes concepts converted into nodes in a vector space. It works in collaboration with a word tokenizer and uses a typical TF Vectorizer to measure the term frequency of the query for a given raw text. 
The SPARQL query retrieves the value for the `dbo:abstract` predicate, which is simply an entity in the DBpedia database. The similarity of words from the query on both resources is measured. 
An example input might be `"Steve Jobs"`, `"Apple Store"`, with the query being `"computers"`. Output -> /screens/Screenshot_4.png

### /api/v1/concepts_similarity/resource1/resource2/
This endpoint analyzes concepts using a vector-based approach for calculating similarity. The BART model is used to generate vectors, and the cosine similarity metric is applied to compare them. 
Based on the two input categories, the results are mapped into pairs, and each result from category 1 is compared to each result from category 2. 
This generates a list of all pairs and returns all possible combinations as a response.

## Application Setup and Usage

To build and run the application, the following is required:

- **Python 3.9 or higher**: Its recommended to use a virtual environment for installing dependencies.
- **Pip**: The tool for managing Python packages.
- Additional packages: FastAPI and Unicorn server.

All required packages are listed in the `requirements.txt` file.

### Build Steps:

Clone the repository:
   
   git clone <https://github.com/laza315/Concept-Similarity-Tool.git>

Set up a virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

Activate the virtual environment:

    source venv/bin/activate  # On Windows: venv\Scripts\activate

Start the FastAPI development server:

    uvicorn main:app --reload