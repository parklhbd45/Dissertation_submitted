import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
from nltk.stem import PorterStemmer
from tqdm import tqdm

nltk.download('punkt', quiet=True)

print("Loading dataset...")
df = pd.read_csv('/kaggle/input/udemyaspect/Udemy_grouped_by_categoryforaspect.csv')
df = df.head(100000)
print(f"Loaded {len(df)} rows.")

ps = PorterStemmer()

keywords = {
    "Instructor": [
        "instructor", "teacher", "lecturer", "professor", "tutor",
        "teaching", "explanation", "explanations"
    ],
    "Design & Material": [
        "content", "material", "lecture", "video",
        "structure", "organized", "curriculum", "syllabus"
    ],
    "Assignment & Tutorial": [
        "assignment", "exercise", "quiz", "project", "task",
        "tutorial", "practice", "homework", "exam"
    ],
    "Financial": [
        "price", "cost", "fee", "expensive", "cheap",
        "affordable", "value", "worth", "refund"
    ],
    "Skill & Career": [
        "skill", "career", "job", "industry", "professional",
        "employment", "resume", "advancement"
    ],
    "Platform & Technical": [
        "platform", "website", "app", "bug",
        "error", "interface", "login", "download"
    ]
}

stemmed_keywords = {aspect: [ps.stem(word) for word in words] for aspect, words in keywords.items()}

def classify_sentence(sentence):
    sentence = sentence.lower()
    stemmed_sentence = ' '.join([ps.stem(word) for word in sentence.split()])
    
    aspect_counts = {aspect: 0 for aspect in keywords.keys()}
    for aspect, words in stemmed_keywords.items():
        aspect_counts[aspect] = sum(bool(re.search(r'\b' + re.escape(word) + r'\b', stemmed_sentence)) for word in words)
    
    max_count = max(aspect_counts.values())
    if max_count == 0:
        return "Ambiguous"
    
    chosen_aspect = max(aspect_counts, key=aspect_counts.get)
    
    return chosen_aspect

def process_comment(comment):
    sentences = sent_tokenize(comment)
    return [(sentence, classify_sentence(sentence)) for sentence in sentences]

tqdm.pandas()
df['processed'] = df['comment'].progress_apply(process_comment)

df_exploded = df.explode('processed')

df_exploded[['sentence', 'aspect']] = pd.DataFrame(df_exploded['processed'].tolist(), index=df_exploded.index)

df_exploded = df_exploded.drop(columns=['processed'])

df_exploded = df_exploded.reset_index(drop=True)

output_file = '/kaggle/working/Udemy_comments_with_sentences_and_aspects.csv'
print(f"Saving results to {output_file}...")
df_exploded.to_csv(output_file, index=False)

