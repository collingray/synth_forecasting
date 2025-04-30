import asyncio
import openai

SYSTEM_PROMPT = """You are a classification assistant. Your task is to determine if a given text sample qualifies as "news".

Definition of "news":
"News" includes current events, reports on recent happenings, but also encompasses broader stories about the world, such as historical accounts, scientific discoveries, cultural trends, and significant societal developments. The key criterion is whether the information presented could be valuable for understanding context and potentially forecasting future events. It is NOT limited to just breaking news or daily headlines. Filter out irrelevant content like personal anecdotes, spam, advertisements, or purely fictional narratives.

Respond with only "YES" or "NO".
YES - The sample contains information valuable for understanding the world and potentially forecasting.
NO - The sample does not meet the criteria.
"""


async def classify_is_news(sample: str, client: openai.AsyncOpenAI) -> bool:
    """
    Classifies a text sample as "news" or "not news" using the OpenAI API.

    Args:
        sample: The text sample to classify.
        client: An initialized AsyncOpenAI client.

    Returns:
        Boolean of whether the sample is news.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample},
            ],
            max_tokens=1,  # Expecting only YES or NO
            temperature=0,  # For deterministic output
            logit_bias={
                31958: 100,  # "YES"
                14695: 100,  # "NO"
                32968: 100,  # " YES"
                9319: 100,  # " NO"
            },
            n=1,
            stop=None,
        )
        result = response.choices[0].message.content.strip().upper()

        if result not in ["YES", "NO"]:
            print(f"Warning: Unexpected response from API: {result}. Defaulting to NO.")
            return False

        return result == "YES"

    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return False


async def main():
    try:
        openai_client = openai.AsyncOpenAI()

        samples = [
            "The stock market experienced a significant downturn yesterday following the release of new inflation data.",
            "My cat fluffy is sleeping on the couch right now, he looks very peaceful.",
            "Archaeologists in Egypt have unearthed a previously unknown tomb dating back to the 18th Dynasty.",
            "Check out this amazing offer! Buy one get one free on all widgets!",
        ]

        tasks = [classify_is_news(sample, openai_client) for sample in samples]
        results = await asyncio.gather(*tasks)

        for i, (sample, is_news) in enumerate(zip(samples, results)):
            print(f"Sample {i+1}: '{sample[:50]}...' -> Is News: {is_news}")

    except openai.AuthenticationError:
        print(
            "OpenAI API key not found or invalid. Please set the OPENAI_API_KEY environment variable."
        )
    except Exception as e:
        print(f"An error occurred during example execution: {e}")


if __name__ == "__main__":
    asyncio.run(main())
