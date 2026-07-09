"""
DecodeLabs - Week 3 - Project 3: AI Recommendation Logic (Tech Stack Recommender)
Fully self-contained. Skills dataset is embedded directly below - no CSV needed.
Run: python Week3.py Python "Cloud Computing" Automation
Requires: pip install scikit-learn
"""

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SKILLS_REQUIRED = 3
TOP_N = 3

# Embedded dataset: job roles mapped to their required skills ("items")
JOB_ROLES = {
    "Data Scientist": "Python SQL Machine Learning Statistics Data Analysis Pandas NumPy Data Visualization",
    "DevOps Engineer": "AWS Docker Kubernetes CI/CD Linux Automation Terraform Cloud Computing",
    "Backend Developer": "Java Python SQL APIs REST Microservices Spring Boot Databases",
    "Frontend Developer": "JavaScript React HTML CSS TypeScript UI Design Web Design Responsive Design",
    "Cloud Architect": "AWS Azure Cloud Computing Networking Security Automation Terraform System Design",
    "Machine Learning Engineer": "Python Machine Learning TensorFlow PyTorch Deep Learning Data Structures Algorithms MLOps",
    "Data Engineer": "Python SQL ETL Data Pipelines Spark Airflow Cloud Computing Data Warehousing",
    "Cybersecurity Analyst": "Security Networking Linux Penetration Testing Risk Assessment Firewalls Python",
    "Mobile App Developer": "Java Kotlin Swift Android iOS Mobile UI APIs Git",
    "Systems Administrator": "Linux Windows Server Networking Automation Scripting Security Git",
    "Full Stack Developer": "JavaScript React Node.js Python SQL APIs HTML CSS Git",
    "QA / Test Automation Engineer": "Python Java Test Automation Selenium CI/CD Git Quality Assurance",
    "Database Administrator": "SQL Database Design Data Warehousing Backup Recovery Performance Tuning Linux",
    "AI Research Engineer": "Python Machine Learning Deep Learning Research Statistics Algorithms PyTorch",
    "Site Reliability Engineer": "Linux Automation Kubernetes Monitoring CI/CD Cloud Computing Scripting",
}


def ingest_user_skills(skills: list[str]) -> str:
    """STEP 1: Ingestion - validate and normalize user skills."""
    if len(skills) < MIN_SKILLS_REQUIRED:
        raise ValueError(
            f"Please provide at least {MIN_SKILLS_REQUIRED} skills (got {len(skills)})."
        )
    return " ".join(skill.lower().strip() for skill in skills)


def score_and_rank(job_roles: dict, user_profile_text: str, top_n=TOP_N):
    """STEPS 2 & 3: Scoring (TF-IDF + cosine similarity) and Sorting."""
    role_names = list(job_roles.keys())
    skill_documents = [job_roles[name].lower() for name in role_names]

    corpus = skill_documents + [user_profile_text]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    item_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]

    scores = cosine_similarity(user_vector, item_vectors)[0]
    ranked = sorted(zip(role_names, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]  # STEP 4: Filtering


def handle_cold_start(scores_present: bool):
    if not scores_present:
        print("\n⚠️  Cold Start detected: none of your skills matched.")
        print("Falling back to trending roles for new users:")
        print("  1. Full Stack Developer")
        print("  2. Data Scientist")
        print("  3. DevOps Engineer")
        return True
    return False


def run_recommender(user_skills: list[str]):
    user_profile_text = ingest_user_skills(user_skills)
    ranked = score_and_rank(JOB_ROLES, user_profile_text)

    if handle_cold_start(any(score > 0 for _, score in ranked)):
        return

    print(f"\nTOP {TOP_N} RECOMMENDED CAREER PATHS")
    print(f"Based on your skills: {', '.join(user_skills)}")
    for rank, (role, score) in enumerate(ranked, start=1):
        print(f"{rank}. {role:<28} — {score*100:.1f}% match")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_skills = sys.argv[1:]
    else:
        input_skills = ["Python", "Cloud Computing", "Automation"]
        print(f"(No skills passed via CLI — running demo with: {input_skills})")

    try:
        run_recommender(input_skills)
    except ValueError as e:
        print(f"Error: {e}")
