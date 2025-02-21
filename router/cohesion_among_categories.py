from fastapi import APIRouter, Depends, Query
from typing import Annotated, Optional
from exceptions import BadInputParsed, NoResultsException
from fastapi_cache.decorator import cache
from Queries import Queries
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache import FastAPICache
from exceptions import MissingInput
from logging import getLogger
from SPARQLWrapper import TURTLE, JSON, N3
from rdflib import Graph, URIRef
from vcm import VectorSpaceModel



logger = getLogger(__name__)


router = APIRouter(
    prefix='/api/v1',
    tags=['resource']
)

_endpoint_url = "https://dbpedia.org/sparql"


cacher = InMemoryBackend()
FastAPICache.init(cacher)



@router.get('/concepts_cohesion/{resource}/{resource2}/{query}', summary='Analysis of concepts converted to nodes in vector space')
@cache(expire=2000)
async def concept_analyzer(
    resource1: str, 
    resource2: str, 
    predictive_query: Annotated[str, Query(max_length=15)], 
    page: int = 1, 
    page_size: Optional[int] = None
):
    if not resource1 or not resource2:
        raise MissingInput

    list_of_inputs = [resource1, resource2]
    aggregate_scrapped = []  
    invalid_resources = []

    for resource in list_of_inputs:
        try:
            q_instance = Queries(url="http://dbpedia.org/", resource=resource)

            if q_instance.dbpedia_uri_validator(resource_obj=resource):
                print(f"{resource} stranica je validna")
                query = q_instance.document_scrapper()
                results = await q_instance.execute_sparql_query(_endpoint_url, query, JSON)

                for res in results['results']['bindings']:
                    lang, value = res['object'].get('xml:lang'), res['object'].get('value', '')
                    if lang == 'en' and value:
                        aggregate_scrapped.append(value)
            else:
                print(f"{resource} stranica nije validna.")
                invalid_resources.append(resource)
            
        except Exception as e:
            print(f"Greska za {resource}: {e}")
            invalid_resources.append(resource)

    if invalid_resources:
        return {"message": f"Neki resursi nisu validni: {', '.join(invalid_resources)}"}

    if len(aggregate_scrapped) < 2:
        return {"error": "Nije pronadjeno dovoljno tekstualnih podataka za poredjenje."}

    similarity_model = VectorSpaceModel(aggregate_scrapped, predictive_query)
    comp_results = similarity_model.concept_comparator()

    return {"similarity_results": comp_results}


@router.get('/concepts_similarity/{resource}/{resource2}/', summary='Analysis of concepts using BART Model for generating vectors and making comparation of their similarities with cosine similarity.')
@cache(expire=2000)
async def concept_similarity(
    resource1: str, 
    resource2: str, 
    page: int = 1, 
    page_size: Optional[int] = None
):
    if not resource1 or not resource2:
        raise MissingInput

    list_of_inputs = [resource1, resource2]
    matrix = []

    for resource in list_of_inputs:
        try:
            q_instance = Queries(url="http://dbpedia.org/", resource=resource)
            query = q_instance.parent_category_Q()

            result = await q_instance.execute_sparql_query(_endpoint_url, query, TURTLE)
            
            G = Graph()
            G.parse(data=result, format="turtle")
            
            print(f"Broj cvorova u grafu za {resource}: {len(G)}")

            lista = []
            for obj in G.objects():
                _concat = str(obj).split("/")[-1]  
                if "Category:" in _concat:
                    obj = _concat.replace("Category:", "") 
                    lista.append(obj)

            source = {f"{resource}": lista}
            matrix.append(source)

        except Exception as e:
            print(f"Greska prilikom procesuiranja resource {resource}: {e}")
            raise e

    if len(matrix) < 2:
        return {"error": "Nema dovoljno podataka za uporedjivanje."}

    lista_parova = []
    first_values = list(matrix[0].values())[0]
    second_values = list(matrix[1].values())[0]

    # documents_to_compare = [
    # ["Barack_Obama", "Politics"],
    # ["Machine_Learning", "Data_Science"]
    # ]

    # ovo mora biti trimovano jer verovatno BART model ne prepoznaje _ i zbog toga su rezultati uvek N/A za resurse koji su validni
    for i in first_values:
        i_clean = i.replace("_", " ")  # Uklanjamo "_"
        for j in second_values:
            j_clean = j.replace("_", " ")  # Uklanjamo "_"
            lista_parova.append([i_clean, j_clean])


    model_bart = VectorSpaceModel(lista_parova, query=None)  
    results_bart = model_bart.concept_similarity()

    return {"results": results_bart}


'''Resource1:Barack Hussein Obama II (/bəˈrɑːk huːˈseɪn oʊˈbɑːmə/ bə-RAHK hoo-SAYN oh-BAH-mə; born August 4, 1961) is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States. 
He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004, and previously worked as a civil rights lawyer before entering politics. Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago. 
In 1988, he enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. After graduating, he became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. 
Turning to elective politics, he represented the 13th district in the Illinois Senate from 1997 until 2004, when he ran for the U.S. Senate. Obama received national attention in 2004 with his March Senate primary win, his well-received July Democratic National Convention keynote address, and his landslide November election to the Senate. 
In 2008, after a close primary campaign against Hillary Clinton, he was nominated by the Democratic Party for president and chose Joe Biden as his running mate. Obama was elected over Republican nominee John McCain in the presidential election and was inaugurated on January 20, 2009. 
Nine months later, he was named the 2009 Nobel Peace Prize laureate, a decision that drew a mixture of praise and criticism. Obama's first-term actions addressed the global financial crisis and included a major stimulus package, a partial extension of George W. Bush's tax cuts, legislation to reform health care, a major financial regulation reform bill, and the end of a major US military presence in Iraq. 
Obama also appointed Supreme Court Justices Elena Kagan and Sonia Sotomayor, the latter of whom became the first Hispanic American on the Supreme Court. His policy against terrorism downplayed Bush's counterinsurgency model, expanding air strikes and making extensive use of special forces, and encouraging greater reliance on host-government militaries. 
He also ordered the counterterrorism raid that killed Osama bin Laden. After winning re-election by defeating Republican opponent Mitt Romney, Obama was sworn in for a second term on January 20, 2013. In his second term, Obama took steps to combat climate change, signing a major international climate agreement and an executive order to limit carbon emissions. 
Obama also presided over the implementation of the Affordable Care Act and other legislation passed in his first term, and he negotiated a nuclear agreement with Iran and normalized relations with Cuba. The number of American soldiers in Afghanistan fell dramatically during Obama's 
second term, though U.S. soldiers remained in Afghanistan throughout Obama's presidency. Obama left office on January 20, 2017, and continues to reside in Washington, D.C. 
His presidential library in Chicago began construction in 2021. During Obama's terms as president, the United States' reputation abroad, as well as the American economy, significantly improved. Rankings by scholars and historians place him among the upper to mid tier of American presidents. 
Since leaving office, Obama has remained active in Democratic politics, including campaigning for candidates in various American elections. Outside of politics, Obama has published three bestselling books: Dreams from My Father (1995), The Audacity of Hope (2006) and A Promised Land (2020).

Resource2:A telephone is a telecommunications device that permits two or more users to conduct a conversation when they are too far apart to be easily heard directly. A telephone converts sound, typically and most 
efficiently the human voice, into electronic signals that are transmitted via cables and other communication channels to another telephone which reproduces the sound to the receiving user. The term is derived from Greek: τῆλε (tēle, far) and φωνή (phōnē, voice), together meaning distant voice. A common short form 
of the term is phone, which came into use early in the telephone's history. In 1876, Alexander Graham Bell was the first to be granted a United States patent for a device that produced clearly intelligible replication of the human voice at a second device. This instrument was further developed by many others, and 
became rapidly indispensable in business, government, and in households. The essential elements of a telephone are a microphone (transmitter) to speak into and an earphone (receiver) which reproduces the voice 
at a distant location. The receiver and transmitter are usually built into a handset which is held up to 
the ear and mouth during conversation. The transmitter converts the sound waves to electrical signals which are sent through the telecommunication system to the receiving telephone, which converts the signals into audible sound in the receiver or sometimes a loudspeaker. Telephones permit transmission in both directions simultaneously. 
Most telephones also contain an alerting feature, such as a ringer or a visual indicator, to announce an incoming telephone call. Telephone calls are initiated most commonly with a keypad or dial, affixed to the telephone, to enter a telephone number, which is the address of the call recipient's telephone in the telecommunication system, but other methods existed in the early history of the telephone. 
The first telephones were directly connected to each other from one customer's office or residence to another customer's location. Being impractical beyond just a few customers, these systems were quickly replaced by manually operated centrally located switchboards. 
These exchanges were soon connected together, eventually forming an automated, worldwide public switched telephone network. For greater mobility, various radio systems were developed for transmission between mobile stations on ships and automobiles in the mid-20th century. 
Hand-held mobile phones were introduced for personal service starting in 1973. In later decades, their analog cellular system evolved into digital networks with greater capability and lower cost. 
Convergence in communication services has provided a broad spectrum of capabilities in cell phones, including mobile computing, giving rise to the smartphone, the dominant type of telephone in the world today.'''


# Input:

# matrix = [
#     {"Barack_Obama": [
#         "21st-century_American_politicians", 
#         "Presidents_of_the_United_States",
#         "Democratic_Party_presidents_of_the_United_States"
#     ]},
#     {"Machine_learning": [
#         "Learning_in_computer_vision",
#         "Semisupervised_learning",
#         "Applied_machine_learning",
#         "Data_mining_and_machine_learning_software",
#         "Computational_learning_theory",
#         "Blockmodeling"
#     ]}
# ]

# Output:

# [
#     ["21st-century_American_politicians", "Learning_in_computer_vision"],
#     ["21st-century_American_politicians", "Semisupervised_learning"],
#     ["21st-century_American_politicians", "Applied_machine_learning"],
#     ["21st-century_American_politicians", "Data_mining_and_machine_learning_software"],
#     ["21st-century_American_politicians", "Computational_learning_theory"],
#     ["21st-century_American_politicians", "Blockmodeling"],

#     ["Presidents_of_the_United_States", "Learning_in_computer_vision"],
#     ["Presidents_of_the_United_States", "Semisupervised_learning"],
#     ["Presidents_of_the_United_States", "Applied_machine_learning"],
#     ["Presidents_of_the_United_States", "Data_mining_and_machine_learning_software"],
#     ["Presidents_of_the_United_States", "Computational_learning_theory"],
#     ["Presidents_of_the_United_States", "Blockmodeling"],

#     ["Democratic_Party_presidents_of_the_United_States", "Learning_in_computer_vision"],
#     ["Democratic_Party_presidents_of_the_United_States", "Semisupervised_learning"],
#     ["Democratic_Party_presidents_of_the_United_States", "Applied_machine_learning"],
#     ["Democratic_Party_presidents_of_the_United_States", "Data_mining_and_machine_learning_software"],
#     ["Democratic_Party_presidents_of_the_United_States", "Computational_learning_theory"],
#     ["Democratic_Party_presidents_of_the_United_States", "Blockmodeling"]
# ]
