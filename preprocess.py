import os
import zipfile
from datasets import load_dataset, DatasetDict, Features, Image, Value, concatenate_datasets
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage


def preprocess_slake():
    print("--- 1. Downloading and Extracting Images ---")
    # Download imgs.zip from the BoKelvin/SLAKE repository
    # We use hf_hub_download because the default load_dataset might not handle the zip automatically
    img_zip_path = hf_hub_download(
        repo_id="BoKelvin/SLAKE",
        filename="imgs.zip",
        repo_type="dataset"
    )

    # Extract images to a local directory
    extract_path = "./slake_images"
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(img_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Images extracted to {extract_path}")
    else:
        print(f"Images already exist at {extract_path}")

    print("\n--- 2. Loading and Filtering Dataset ---")
    # Load the text data (JSON files are usually loaded automatically by load_dataset)
    # SLAKE has 'train', 'validation', and 'test' splits by default
    dataset = load_dataset("BoKelvin/SLAKE")

    # Concatenate all splits to apply a global filter and new split ratio
    combined_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    # Filter for English language only ('q_lang' == 'en')
    # Note: 'q_lang' is the column name for language in SLAKE
    english_dataset = combined_dataset.filter(lambda example: example['q_lang'] == 'en')
    print(f"Filtered English samples: {len(english_dataset)}")

    print("\n--- 3. Reformatting to Match VQA-RAD ---")

    # Define the image loading function
    def add_image_column(example):
        # The 'img_name' in SLAKE is like 'xmlab1/source.jpg'
        # We assume the zip extraction maintained this structure inside 'slake_images'
        # Sometimes SLAKE zip extracts into a subfolder, we handle standard extraction here.
        # Check if the path exists directly or inside a 'imgs' folder (common in some zips)

        image_path = os.path.join(extract_path, example['img_name'])

        # Fallback: sometimes zips extract to a root folder name
        if not os.path.exists(image_path):
            # Try checking if there's an intermediate folder (e.g., 'imgs')
            image_path = os.path.join(extract_path, 'imgs', example['img_name'])

        try:
            img = PILImage.open(image_path).convert("RGB")
            return {"image": img}
        except Exception as e:
            return {"image": None}

    # Apply the image loading
    formatted_dataset = english_dataset.map(add_image_column)

    # Filter out any images that failed to load
    formatted_dataset = formatted_dataset.filter(lambda x: x['image'] is not None)

    # Select and Rename columns to match VQA-RAD: feature columns are 'image', 'question', 'answer'
    # SLAKE columns: 'question', 'answer' exist. 'image' is created above.
    formatted_dataset = formatted_dataset.select_columns(['image', 'question', 'answer'])

    # Enforce features to ensure they match VQA-RAD types exactly
    # VQA-RAD features: image (Image), question (string), answer (string)
    features = Features({
        'image': Image(),
        'question': Value('string'),
        'answer': Value('string')
    })
    formatted_dataset = formatted_dataset.cast(features)

    print("\n--- 4. Splitting Dataset (70/10/20) ---")
    # We want Train: 70%, Val: 10%, Test: 20%

    # First, split off the Test set (20%)
    # remaining_dataset will be 80%
    test_split = formatted_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = test_split['test']
    remaining_dataset = test_split['train']

    # Now split the remaining 80% into Train and Val
    # We want Val to be 10% of the ORIGINAL total.
    # Since remaining is 80% (0.8) of total, Val should be 0.1 / 0.8 = 0.125 of remaining.
    val_split = remaining_dataset.train_test_split(test_size=0.125, seed=42)

    train_dataset = val_split['train']
    val_dataset = val_split['test']

    # Create the final DatasetDict
    final_dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    print("Final Splits:")
    print(f"Train: {len(final_dataset['train'])} ({len(final_dataset['train']) / len(formatted_dataset):.1%})")
    print(
        f"Val:   {len(final_dataset['validation'])} ({len(final_dataset['validation']) / len(formatted_dataset):.1%})")
    print(f"Test:  {len(final_dataset['test'])} ({len(final_dataset['test']) / len(formatted_dataset):.1%})")

    print("\nSample Element:")
    print(final_dataset['train'][0])

    return final_dataset


if __name__ == "__main__":
    # Execute the preprocessing
    slake_vqa_rad_style = preprocess_slake()

    # Optional: Save to disk
    slake_vqa_rad_style.save_to_disk("slake_vqa_rad_format")
