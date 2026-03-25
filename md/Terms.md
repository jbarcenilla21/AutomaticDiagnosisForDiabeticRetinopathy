# Terms

## 1. Evaluation Criteria

The evaluation is done through a **Codabench challenge** in two submission categories:

| Category | Description |
|----------|-------------|
| **Custom** | Network built from scratch. No external networks, packages, or pretrained models allowed. Full code must appear in the notebook. |
| **Fine-tuning** | Network pretrained on another dataset and fine-tuned for DR. Models from `torchvision` or external sources are allowed. |

### Grading Breakdown (out of 10)

| Component | Points |
|-----------|--------|
| Technical content of the report | 4 |
| Competition results on Codabench (linear regression between best and worst scores) | 4 |
| Quality of submitted Python code | 2 |

### Report Format

A brief report of **2 sides**:
- 1 side — description of the proposed solutions
- 1 side — extra material (tables, figures, references)

The report should list and briefly discuss the key decisions and extensions made when optimizing the system. It does not need to cover every change in detail.

> **Important:** Each group member must focus on specific improvements or experiments, and the report must clearly indicate individual contributions.

---

## 2. Challenge Rules (Codabench)

| Rule | Detail |
|------|--------|
| **Timeline** | Wednesday March 25 → Tuesday April 21 |
| **Daily submission limit** | 4 solution files per day |
| **Total submission limit** | 100 submissions |
| **Code verification** | Code will be reviewed to verify results. Non-reproducible results → disqualification |

---

## 3. Submission Details

### Codabench uploads

Two `.csv` files (one per category), each containing:
- A **1000×1** matrix with the DR score for each test image
- One number per row, plain text format

### Aula Global ZIP file

The ZIP must include:

1. The two `.csv` files with test outputs (same format as above)
2. The report (as described in Section 1)
3. A single notebook integrating both models (Custom and Fine-tuning)

---

**Submission deadline: Tuesday, April 21 at 11:59 PM**
