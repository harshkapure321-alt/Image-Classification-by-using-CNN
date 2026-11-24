"""
Downloads authentic animal images for Mammals, Birds, and Reptiles
using fast batch queries to Wikipedia/Wikimedia.
"""

import os
import requests
from PIL import Image
import io

HEADERS = {
    "User-Agent": "AnimalClassifierBot/2.0 (Mozilla/5.0; Windows NT 10.0; Win64; x64) Educational/Research"
}

BATCH_TARGETS = {
    "mammals": [
        "Blue_whale", "Humpback_whale", "Sperm_whale", "Fin_whale", "Beluga_whale",
        "Orca", "Bottlenose_dolphin", "African_elephant", "Asian_elephant", "Lion",
        "Tiger", "Leopard", "Cheetah", "Grizzly_bear", "Polar_bear",
        "Giant_panda", "Giraffe", "Hippopotamus", "Rhinoceros", "Chimpanzee",
        "Kangaroo", "Red_fox", "Gray_wolf", "Horse", "Zebra"
    ],
    "birds": [
        "Bald_eagle", "Golden_eagle", "Peregrine_falcon", "Red-tailed_hawk", "Barn_owl",
        "Great_horned_owl", "King_penguin", "Emperor_penguin", "Atlantic_puffin", "Mallard",
        "Mute_swan", "Greater_flamingo", "Scarlet_macaw", "Blue-and-yellow_macaw", "Toco_toucan",
        "Common_ostrich", "Emu", "Peafowl", "Hummingbird", "Common_raven",
        "Blue_jay", "American_robin", "Pileated_woodpecker", "Pelican", "Kingfisher"
    ],
    "reptiles": [
        "Crocodile", "Alligator", "Komodo_dragon", "Iguana", "Chameleon",
        "Gecko", "Snake", "Cobra", "Viper", "Turtle",
        "Tortoise", "Lizard", "Gharial", "Gila_monster", "Skink",
        "Leatherback_sea_turtle", "Tuatara", "Rattlesnake", "Boa_constrictor", "Anaconda"
    ]
}


def download_batch_for_class(class_name, titles, base_dir="dataset"):
    folder = os.path.join(base_dir, class_name)
    os.makedirs(folder, exist_ok=True)
    
    titles_str = "|".join(titles)
    api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={titles_str}&prop=pageimages&pithumbsize=300&format=json"

    print(f"\n[Fetching metadata] Querying {len(titles)} titles for class: '{class_name}'...")
    try:
        r = requests.get(api_url, headers=HEADERS, timeout=12)
        pages = r.json().get("query", {}).get("pages", {})
    except Exception as e:
        print(f"Error querying API for {class_name}: {e}")
        return

    saved_count = 0
    for pid, page in pages.items():
        title = page.get("title", "")
        thumb = page.get("thumbnail", {}).get("source")
        if not thumb:
            continue

        safe_name = title.lower().replace(" ", "_").replace("-", "_") + ".jpg"
        save_path = os.path.join(folder, safe_name)

        if os.path.exists(save_path):
            saved_count += 1
            continue

        try:
            img_resp = requests.get(thumb, headers=HEADERS, timeout=10)
            if img_resp.status_code == 200:
                img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                img.save(save_path, "JPEG", quality=90)
                saved_count += 1
                print(f"  [OK] Saved: {class_name}/{safe_name}")
        except Exception as e:
            print(f"  [Error] {title}: {e}")

    print(f"Total images in '{class_name}': {len(os.listdir(folder))}")


def main():
    print("=" * 60)
    print(" Downloading Animal Classes: Mammals, Birds, Reptiles ")
    print("=" * 60)

    for class_name, titles in BATCH_TARGETS.items():
        download_batch_for_class(class_name, titles)

    print("\nDataset ready!")


if __name__ == "__main__":
    main()
