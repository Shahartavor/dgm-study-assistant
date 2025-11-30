from pathlib import Path

def rename_slide_images():
    base_dir = Path("src/dgm_study_assistant/rag/data/gen_slides_images/DGM_L6_Variational_Inference")

    for img_path in base_dir.glob("*.png"):
        old_name = img_path.name  # e.g. "7 Normalizing Flows_page_0.png"

        # Extract page number from old name
        # It’s always the last chunk before ".png"
        # E.g. "7 Normalizing Flows_page_12.png"
        new_name = old_name.replace("DGM_L8", "DGM_L6")
        new_path = img_path.parent / new_name

        print(f"  {old_name} -> {new_name}")
        img_path.rename(new_path)

    print("\n✔ All images renamed successfully!")

if __name__ == "__main__":
    rename_slide_images()
