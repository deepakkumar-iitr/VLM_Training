#  Step 1 Add a FilteredDataset wrapper
# ============ Safe Dataset Wrapper ============
class FilteredDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.valid_indices = []
        for i in range(len(base_dataset)):
            try:
                sample = base_dataset[i]
                if all(k in sample for k in ['image', 'question', 'answers']):
                    self.valid_indices.append(i)
            except Exception as e:
                continue  # skip bad samples

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.valid_indices[idx]]

## Step 2 Replace dataset loading with filtering
##Replace below lines 
train_dataset = DocVQADataset("train.csv", "train_images")
val_dataset = DocVQADataset("val.csv", "val_images")
#With
train_dataset = FilteredDataset(DocVQADataset("train.csv", "train_images"))
val_dataset = FilteredDataset(DocVQADataset("val.csv", "val_images"))

## Step 3 Add Safe collate_fn
#  Also update your collate_fn with batch checking:
# ============ Collate Function ============
def collate_fn(batch):
    valid_batch = [item for item in batch if all(k in item for k in ["image", "question", "answers"])]
    
    if len(valid_batch) == 0:
        raise ValueError("All items in batch were invalid or missing keys.")

    pixel_values = torch.stack([item["image"] for item in valid_batch])
    questions = [item["question"] for item in valid_batch]
    answers = [item["answers"] for item in valid_batch]  # List[List[str]]

    return {
        "pixel_values": pixel_values,
        "questions": questions,
        "answers": answers
    }
####  Optional Debug Tip
sample = train_dataset[0]
print("Sample keys:", sample.keys())
print("Image shape:", sample["image"].shape)
print("Question:", sample["question"])
print("Answers:", sample["answers"])
