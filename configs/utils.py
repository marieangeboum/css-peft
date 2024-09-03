import os
import yaml
import json
import torch
import random
from collections import defaultdict


def cumuler_valeurs(d):
    total = 0
    for cle, valeur in d.items():
        if cle >= 12:
            total += valeur
    cles_a_supprimer = [cle for cle in d.keys() if cle >= 12]
    for cle in cles_a_supprimer:
        del d[cle]
    d[12] = round(total,2)
    return d


def select_random_domains(directory, num_domains, seed=None):
    if seed is not None:
        random.seed(seed)
    domain_list = os.listdir(directory)
    selected_domains = random.sample(domain_list, num_domains)
    random.shuffle(selected_domains)
    return selected_domains

def load_json_file(file_path):
    with open(file_path, 'r') as f : 
        metadata  = json.load(f)
    return metadata

def load_config_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_domain_sequence(config_file, new_domain_sequence, field = "domain_sequence"):
    config = load_config_yaml(config_file)
    config["dataset"]["flair1"][field] = new_domain_sequence
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def domain_class_weights(metadatafile, domains, binary_label = None, binary = False):
    metadata = load_json_file(metadatafile) 
    nombre_domaines = len(domains)
    # Dictionnaire pour stocker les valeurs de label pour chaque domaine spécifié
    valeurs_label_par_domaine = {}
    domain_class_weights = {}
    # Parcours de chaque domaine spécifié
    for domaine_specifie in domains:
        valeurs_label = []
        # Parcours de chaque élément du dictionnaire
        for key, value in metadata.items():
            # Vérification du domaine
            if value.get("domain") == domaine_specifie:
                # Récupération de la valeur de la clé "label"
                label_value = value.get("labels")
                if label_value:
                    valeurs_label.append(label_value)
        # Stockage des valeurs de label pour le domaine spécifié dans le dictionnaire
        valeurs_label_par_domaine[domaine_specifie] = valeurs_label
        
        # Dictionnaire pour stocker les occurrences de chaque label unique et la somme de leurs valeurs
        occurrences_et_sommes = defaultdict(lambda: {"occurrences": 0, "somme": 0})
        
        # Parcours de chaque ensemble de données
        for d in valeurs_label_par_domaine[domaine_specifie]:
            for label, valeur in d.items():
                occurrences_et_sommes[int(label)-1]["occurrences"] += 1
                occurrences_et_sommes[int(label)-1]["somme"] += valeur
        
        # Dictionnaire pour stocker les proportions pour chaque label unique
        proportions_par_label = {}
        # Calcul des proportions pour chaque label unique
        for label, info in occurrences_et_sommes.items():
            occurences = info["occurrences"]
            somme = info["somme"]
            proportions_par_label[label] = round(somme / occurences, 2) if occurences > 0 else 0
        domain_class_weights[domaine_specifie] = proportions_par_label
        proportions_cumulatives = {}
        
        for domaine, proportions in  domain_class_weights.items():
            for classe, proportion in proportions.items():
                if classe not in proportions_cumulatives:
                    proportions_cumulatives[classe] = round(proportion / nombre_domaines,2)
                else:
                    proportions_cumulatives[classe] += round(proportion / nombre_domaines,2)
    domain_class_weights = {cle: dict(sorted(v.items())) for cle, v in domain_class_weights.items()}
    domain_class_weights = {cle: cumuler_valeurs(d.copy()) for cle, d in domain_class_weights.items()}
    
    proportions_cumulatives = {cle: proportions_cumulatives[cle] for cle in sorted(proportions_cumulatives.keys())}
    proportions_cumulatives = cumuler_valeurs(proportions_cumulatives)
    
    
    if binary : 
        binary_values =  {key: value[binary_label] for key, value in domain_class_weights.items() if binary_label in value}
        sum_binary_values =  sum(binary_values.values())/nombre_domaines
        
        domain_class_weights = binary_values
        proportions_cumulatives = sum_binary_values
    # domain_class_weights = {domaine: torch.tensor(list(valeurs.values())) for domaine, valeurs in domain_class_weights.items()}
    # proportions_cumulatives = torch.tensor(list(proportions_cumulatives.values()))
    all_class_weights = proportions_cumulatives
    # all_class_weights = list(all_class_weights.values())
    # domain_class_weights = list(domain_class_weights.values)
    return domain_class_weights, all_class_weights
            

def common_keys(data,cumul_data, threshold = 6):
    # Obtenir les clés non nulles pour le premier domaine
    keys_non_null = set(data[next(iter(data))].keys())

    # Filtrer les clés non nulles et supérieures à 8% dans tous les domaines
    for domain_data in data.values():
        keys_non_null &= {key for key, value in domain_data.items() if value > threshold}
    values_keys_non_null = {key: cumul_data[key]/100 for key in keys_non_null}
    # Afficher les clés non nulles communes à tous les domaines
    return  values_keys_non_null
