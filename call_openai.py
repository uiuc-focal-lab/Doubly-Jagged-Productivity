from openai import OpenAI



client = OpenAI()  # reads OPENAI_API_KEY from env



resp = client.responses.create(

            model="gpt-5",

                input="Say hi in one sentence.",

                    max_output_tokens=128,

                        )



print(resp.output_text)


