
languages_groups = [
    ["en"],
    ["de", "nl", "sv", "da", "af"],
    ["fr", "es", "it", "pt", "ro"],
    ["ru", "cs", "pl", "bg", "uk"],
    ["id", "jv", "ms", "tl"],
    ["ja", "zh", "ko", "vi"],
]
languages_families = {
    "English": ["en"],
    "Germanic": ["de", "nl", "sv", "da", "af"],
    "Romance": ["fr", "es", "it", "pt", "ro"],
    "Slavic": ["ru", "cs", "pl", "bg", "uk"],
    "Malayo-Polyn.": ["id", "jv", "ms", "tl"],
    "Others": ["ja", "zh", "ko", "vi"],
}
languages = [item for sublist in languages_groups for item in sublist]
bridge_languages = [
    ["en"],
    ["de", "nl"],
    ["fr", "es"],
    ["ru", "cs"],
    ["id"],
    ["ja", "zh"],
]
all_bridge_languages = [item for sublist in bridge_languages for item in sublist]

# lack a bridge language in Malayo-Polyn., so we connect non-bridge languages with other families.
special_pairs = [
    "de-ms", "fr-ms", "ru-ms", "vi-ms", "ja-ms", "zh-ms", "jv-ms",
    "ja-tl", "zh-tl", "vi-tl", "ru-tl", "fr-tl", "de-tl",
]
total_pairs=[]
bidirection_pairs=[]

def count_total_pairs():
    global total_pairs
    # English-centric
    english_centric_pairs = []
    for idx, lang in enumerate(languages):
        if idx == 0: continue
        english_centric_pairs.append(f"en-{lang}")
    
    # group-inner pairs
    # for a language in this group, it will connect to all bridge of this group
    # skip bridge language in this step
    group_inner_pairs = []
    for idx, gourp in enumerate(languages_groups):
        if idx == 0: continue
        bridge_languages_for_this_group = bridge_languages[idx]
        for bridge in bridge_languages_for_this_group:
            for group_lang in gourp:
                if group_lang in bridge_languages_for_this_group: continue
                group_inner_pairs.append(f"{bridge}-{group_lang}")
    
    # group-outer pairs
    # bridge languages will be fully connected
    group_outer_pairs = []
    for i in range(len(all_bridge_languages)):
        # skip English
        if i == 0: continue
        for j in range(i + 1, len(all_bridge_languages)):
            group_outer_pairs.append(f"{all_bridge_languages[i]}-{all_bridge_languages[j]}")
         
    total_pairs = english_centric_pairs + group_inner_pairs + group_outer_pairs + special_pairs
    return total_pairs

def count_bidirection_pairs():
    global bidirection_pairs
    global total_pairs
    if len(total_pairs) == 0:
        total_pairs = count_total_pairs()
    inverse_pairs = []
    for pair in total_pairs:
        src, tgt = pair.split("-")
        inverse_pairs.append(f"{tgt}-{src}")
    bidirection_pairs = total_pairs + inverse_pairs
    return bidirection_pairs

def get_total_pairs():
    return total_pairs if len(total_pairs) > 0 else count_total_pairs()

def get_bidirection_pairs():
    return bidirection_pairs if len(bidirection_pairs) > 0 else count_bidirection_pairs()