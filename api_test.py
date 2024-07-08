import praw

def main():
    reddit = praw.Reddit(
        client_id='ZC5eI8EbdWOlcTQU1C7cCg',  # Asegúrate de que esto es correcto
        client_secret='a1MvR2C_syEBqKPWC75HRb-li28jSQ',  # Asegúrate de que esto es correcto
        user_agent='python:praw:example_app:v1.0 (by /u/Suspicious_Sport2182)'  # Describe tu aplicación
    )

    subreddit = reddit.subreddit('soccer')
    
    print("Mostrando los 10 posts más populares en r/soccer:")
    for submission in subreddit.hot(limit=10):
        print(f"Titulo: {submission.title}")
        print(f"URL: {submission.url}")
        print(f"Puntuación: {submission.score}")

if __name__ == "__main__":
    main()
