# Hate Target Detection Feature

## Overview
Added hate target detection and display functionality to identify and show the specific target of hateful content (e.g., Religion, Race, Gender, etc.).

## Backend Changes

### 1. `multi_model_classifier.py`
**New fields in classification output:**
- `hate_target`: The identified target of hate (e.g., "Religion", "Race", "Gender", "Disability")
- `hate_target_confidence`: Confidence score for the target classification (0.0-1.0)

**Logic:**
```python
# Only extract target if content is toxic (overall_toxicity > 0.5)
if overall_toxicity > threshold and target_scores:
    # Find highest confidence target (excluding negative labels)
    positive_targets = {k: v for k, v in target_scores.items() 
                       if 'not' not in k.lower() and v > 0.3}
    
    if positive_targets:
        hate_target = max(positive_targets, key=positive_targets.get)
        # Clean up label: "religion_based" -> "Religion"
```

**Target categories detected:**
- Religion-based hate
- Race-based hate
- Gender-based hate
- Disability-based hate
- Age-based hate
- Ethnicity-based hate
- Other categories from the BERT multilabel model

### 2. `extreme.py`
**Added to segment response:**
```python
if classification.get("hate_target"):
    seg_data["hateTarget"] = classification["hate_target"]
    seg_data["hateTargetConfidence"] = classification["hate_target_confidence"]
```

## Frontend Changes

### `+page.svelte`
**Updated Segment type:**
```typescript
type Segment = {
  startTime: TimeStamp;
  endTime: TimeStamp;
  text: string;
  extreme: number;
  hateTarget?: string;           // NEW
  hateTargetConfidence?: number; // NEW
};
```

**Visual indicators:**
- **⚠️ Warning icon**: Appears as superscript after hateful segments
- **Enhanced tooltip**: Shows target and confidence when hovering
- **Red color**: Background color intensity based on toxicity level

**Example display:**
```
"You are a terrible person"⚠️
```

Tooltip shows:
```
00:15-00:18: 87.5%
Target: Religion (75.2%)
```

## API Response Example

```json
{
  "segments": [
    {
      "text": "hateful content here",
      "startTime": {"minute": 0, "second": 15},
      "endTime": {"minute": 0, "second": 18},
      "extreme": 0.875,
      "hateTarget": "Religion",
      "hateTargetConfidence": 0.752,
      "classification": {
        "overall_toxicity": 0.875,
        "is_toxic": true,
        "hate_target": "Religion",
        "hate_target_confidence": 0.752,
        "model_outputs": {...}
      }
    }
  ]
}
```

## Model Used
**wesleyacheng/hate-speech-multilabel-classification-with-bert**
- Multi-label BERT classifier
- Trained on hate speech dataset with target labels
- Provides confidence scores for multiple target categories

## Features
✅ Only shows target when content is toxic (>0.5 toxicity)
✅ Filters out low-confidence predictions (<0.3)
✅ Clean, human-readable labels
✅ Visual warning indicators in frontend
✅ Detailed tooltip information
✅ Confidence scores for transparency

## Usage
The hate target is automatically detected and displayed when:
1. Content is classified as toxic (overall_toxicity > 0.5)
2. Target classifier has confidence > 0.3
3. Target is a positive label (not "not_cyberbullying" etc.)

No additional configuration needed - works out of the box!
