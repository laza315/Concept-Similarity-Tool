from fastapi import APIRouter, Depends
from typing import Optional
from rdflib import Graph, URIRef
from exceptions import BadInputParsed, NoResultsException
import queue
from Queries import Queries
from typing import List
from configparser import ConfigParser
import asyncio
from fastapi_cache.decorator import cache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache import FastAPICache
from logging import getLogger
from SPARQLWrapper import TURTLE, JSON

logger = getLogger(__name__)

config = ConfigParser()
config.read('config.ini')
# print(config['endpoint']['sparkQL_url'])

router = APIRouter(
    prefix='/api/v1',
    tags=['resource']
)

_endpoint_url = "https://dbpedia.org/sparql"


cacher = InMemoryBackend()
FastAPICache.init(cacher)


    
def remove_unused_prefixes(self):
        """ Clean up the long list of namespaces rdflib produces."""
        context = dict(self.graph.namespaces())
        used_prefixes = ['rdf', 'rdfs', 'sdo', 'skos', 'xml', 'xsd']
        return {i: j for i, j in context.items() if i in used_prefixes}




@router.get('/categories/{resource}', summary='Retrieve all parent and child categories for a given resource')
@cache(expire=3600)
async def get_slave_and_parent_categories_for_resource(resource: str, page: int = 1, page_size: Optional[int] = None):
    '''Afunkcija za izvlacenje hijerarhije kategorija za resurs i rekurzivno izvlacenje istih za svaki dobijeni rezultat u Grafu. '''
    logger.info('message from resource module')
    # ulazimo sa 'Barack_Obama' test inputom
    q_instance = Queries(url="http://dbpedia.org/", resource=resource)
    query = q_instance.parent_category_Q()
    # Trenutno, posto cemo ici po BFS alogirtmu, u prvom searchu po okidanju API-ja u graf dodajemo prvi layer oko resursa prvom garniturom kategorija, sto nam daje dubinu 1
    result = await q_instance.execute_sparql_query(_endpoint_url, query, TURTLE) 

    g = Graph()
    graph = g.parse(data=result, format='turtle') # smestamo ga u graf
    print("Broj cvorova u grafu:", len(graph))

    visited_categories = set()
    lista_cekanja = queue.Queue()
    # krecemo od node-a jer moramo pokrenuti loop ali je odmah markirana kao visited ,
    # tako da se nece izvrsiti dupla pretraga njenih kategorija
    lista_cekanja.put(resource) 
    visited_categories.add(resource)


    layers = [] # inicijalizacija layer-a sto predstavlja jedan layer svih kategorija oko jednog resursa u grafu
    depth = 0 # pracenje dubine grafa. Od prvog noda do zadnjeg layer-a

    while not lista_cekanja.empty():
        num_nodes_current_level = lista_cekanja.qsize() # approx
        new_categories = []

        tasks = []

        for _ in range(num_nodes_current_level):  
            category = lista_cekanja.get() # uzimamo root iz liste i dekorisemo sa tim stringom/ npr https://dbpedia.org/page/Category:Barack_Obama
            category_uri = URIRef(f"http://dbpedia.org/resource/Category:{category}") # ovime smo samo unapred izvukli sve povezane URIRef cvorove glavnog noda u RDF grafu           
            tasks.append(background_subcategory_digger(category)) # u pozadini pokrecemo izvrsavanje pretrage za taj resurs i da dobijemo sve podkategorije

        subcategory_results = await asyncio.gather(*tasks) # sacekaj da svi taskovi zavrse da bi tek onda mogao da dodas sve odjednom u graf a ne po svakom tasku

        for result in subcategory_results:
            graph += Graph().parse(data=result, format='turtle')  
            
            # Radimo konkatizaciju rezultata jer je objekat u formi linka. 
            # Nama treba samo cist 'Category_Name' za sledecu pretragu
            for obj in graph.objects(subject=category_uri):
                _concat = str(obj).split("/")[-1] 
                if "Category:" in _concat:
                    obj = _concat.replace("Category:", "")

                    # dodajemo sve objekte u nove kategorije za pretragu
                    if obj not in visited_categories:
                        visited_categories.add(obj)
                        new_categories.append(obj)

        # kada je zavrsena pretraga svih kategorija i subkategorija dodaj u graf i prosiri layer listu za taj set rezultata 
        # npr jedna kategorija je imala 3 subkategorije i ovo su rezultati tih kategorija, premda rezultati nemaju svoje podkategorije 
        # [["a", "b", "c"], ["4", "5"], ["char"]]. Pretaga se ovim zavrsila i zadnji nodovi "a,b,c,4,5,char" cine 1 layer oko svog parenta u RDF grafu
        if new_categories:
            expanded_categories = await expand_the_graph(graph, new_categories)
            layers.extend(expanded_categories)
            logger.info(msg=f"Expanded categories at depth {depth + 1}: {expanded_categories}")

            print(f"Expanded categories at depth {depth + 1}: {expanded_categories}")


        for cat in new_categories:  
            lista_cekanja.put(cat)

        if new_categories:  
            depth += 1
            print("Nova tura, nova garnitura")


        layers.extend(new_categories) # za objedinjavanje rezultata za vise kategorija

    return {"layers_depth": depth, "number_of_nodes": len(layers),'categories': layers}

async def background_subcategory_digger(category_val):
    q_instance = Queries(url="http://dbpedia.org/", resource=category_val)
    query = q_instance.parent_category_Q()
    result = await q_instance.execute_sparql_query(_endpoint_url, query, TURTLE)
    return result



async def expand_the_graph(graph: Graph, categories: List[str]) -> List[str]:
    '''BFS za dodavanje kategorija u graf'''
    # slikovito ovako radi:
    '''
    Za root node -> 30 kategorija. Uzimamo prvu kategoriju i idemo u sirinu(dubinu). 
    Ako ona ima svoj child, cuvamo njega i pitamo ima li on? Dok god ima, cuvamo i pakujemo u listu. 
    Time smo zavrsili 1 layer ostali su jos 29 da ispitamo... etc. 
    Svaki node moze biti posecen samo jednom i jednom u grafu!
    '''
    # param: categories je lista kategorija za slaganje

    visited = set(categories)
    stack = queue.Queue(maxsize=5000)

    for category in categories:
         stack.put(category)

    order = []

    while not stack.empty():
        node = stack.get()

        if node not in visited:
            order.append(node)
            visited.add(node)
            
            for n in graph.objects(subject=URIRef(node)): #proveravamo sve URIRef susede glavnog noda u RDF grafu
                if n not in visited:
                     stack.put(str(n))
    
    return order # vracamo layer podataka oko grafa


# {
#   "categories": "@prefix dbc: <http://dbpedia.org/resource/Category:> .\n@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n\ndbc:Machine_learning skos:broader dbc:Artificial_intelligence,\n        dbc:Learning ;\n    skos:narrower dbc:Applied_machine_learning,\n        dbc:Artificial_intelligence_conferences,\n        dbc:Artificial_neural_networks,\n        dbc:Bayesian_networks,\n        dbc:Blockmodeling,\n        dbc:Classification_algorithms,\n        dbc:Cluster_analysis,\n        dbc:Computational_learning_theory,\n        dbc:Data_mining_and_machine_learning_software,\n        dbc:Datasets_in_machine_learning,\n        dbc:Deep_learning,\n        dbc:Dimension_reduction,\n        dbc:Ensemble_learning,\n        dbc:Evolutionary_algorithms,\n        dbc:Genetic_programming,\n        dbc:Inductive_logic_programming,\n        dbc:Kernel_methods_for_machine_learning,\n        dbc:Latent_variable_models,\n        dbc:Learning_in_computer_vision,\n        dbc:Log-linear_models,\n        dbc:Loss_functions,\n        dbc:Machine_learning_algorithms,\n        dbc:Machine_learning_researchers,\n        dbc:Machine_learning_task,\n        dbc:Markov_models,\n        dbc:Natural_language_processing_researchers,\n        dbc:Ontology_learning_\\(computer_science\\),\n        dbc:Reinforcement_learning,\n        dbc:Semisupervised_learning,\n        dbc:Signal_processing_conferences,\n        dbc:Statistical_natural_language_processing,\n        dbc:Structured_prediction,\n        dbc:Supervised_learning,\n        dbc:Support_vector_machines,\n        dbc:Unsupervised_learning .\n\n"
# }



# obj je http://dbpedia.org/resource/Category:Log-linear_models
# All cleaned categories ['Learning_in_computer_vision', 'Artificial_neural_networks', 'Ontology_learning_(computer_science)', 'Markov_models', 'Datasets_in_machine_learning', 'Blockmodeling', 'Genetic_programming', 'Machine_learning_algorithms', 'Machine_learning_researchers', 'Learning', 'Deep_learning', 'Inductive_logic_programming', 'Classification_algorithms', 'Evolutionary_algorithms', 'Kernel_methods_for_machine_learning', 'Artificial_intelligence_conferences', 'Structured_prediction', 'Cluster_analysis', 'Supervised_learning', 'Bayesian_networks', 'Artificial_intelligence', 'Machine_learning_task', 'Data_mining_and_machine_learning_software', 'Support_vector_machines', 'Dimension_reduction', 'Reinforcement_learning', 'Computational_learning_theory', 'Ensemble_learning', 'Loss_functions', 'Signal_processing_conferences', 'Unsupervised_learning', 'Applied_machine_learning', 'Natural_language_processing_researchers', 'Latent_variable_models', 'Semisupervised_learning', 'Statistical_natural_language_processing', 'Log-linear_models']


# obj je http://dbpedia.org/resource/Category:Statistical_natural_language_processing
# All cleaned categories ['Learning_in_computer_vision', 'Artificial_neural_networks', 'Ontology_learning_(computer_science)', 'Markov_models', 'Datasets_in_machine_learning', 'Blockmodeling', 'Genetic_programming', 'Machine_learning_algorithms', 'Machine_learning_researchers', 'Learning', 'Deep_learning', 'Inductive_logic_programming', 'Classification_algorithms', 'Evolutionary_algorithms', 'Kernel_methods_for_machine_learning', 'Artificial_intelligence_conferences', 'Structured_prediction', 'Cluster_analysis', 'Supervised_learning', 'Bayesian_networks', 'Artificial_intelligence', 'Machine_learning_task', 'Data_mining_and_machine_learning_software', 'Support_vector_machines', 'Dimension_reduction', 'Reinforcement_learning', 'Computational_learning_theory', 'Ensemble_learning', 'Loss_functions', 'Signal_processing_conferences', 'Unsupervised_learning', 'Applied_machine_learning', 'Natural_language_processing_researchers', 'Latent_variable_models', 'Semisupervised_learning', 'Statistical_natural_language_processing']