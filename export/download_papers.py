import urllib.request
import os

papers = {
    "Localizing_Persona_Representations.pdf": "https://arxiv.org/pdf/2505.24539.pdf",
    "Geometry_of_Refusal_Concept_Cones.pdf": "https://arxiv.org/pdf/2502.17420.pdf",
    "Contrastive_Activation_Engineering_CAE.pdf": "https://arxiv.org/pdf/2505.06822.pdf",
    "Platonic_Representation_Hypothesis.pdf": "https://arxiv.org/pdf/2405.07987.pdf",
    "Anti_Abliteration.pdf": "https://arxiv.org/pdf/2505.19056.pdf",
    "COSMIC_Refusal_Direction.pdf": "https://arxiv.org/pdf/2505.30006.pdf"
}

export_dir = r"G:\Surper_GCG\poc\export"
os.makedirs(export_dir, exist_ok=True)

for name, url in papers.items():
    filepath = os.path.join(export_dir, name)
    print(f"Downloading {name} from {url}...")
    try:
        # User-Agent is sometimes required by arXiv to avoid blocking 403
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
        print(f"  -> Saved {name}")
    except Exception as e:
        print(f"  -> Failed to download {name}: {e}")

print("Done downloading papers.")
