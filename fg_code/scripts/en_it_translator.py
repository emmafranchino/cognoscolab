client = LMStudioChatClient(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

words = read_words_from_excel(INPUT_EXCEL)
print(f"Loaded {len(words)} rows from: {INPUT_EXCEL}")

results = []
for w in tqdm(words, desc="Translating", unit="word"):
    try:
        row = translate_word(client, w)
    except Exception as e:
        row = {"english": str(w), "italian": "", "note": f"error:{type(e).__name__}: {e}"}
    results.append(row)

out_df = write_results_to_excel(results, OUTPUT_EXCEL)
print(f"Saved {len(out_df)} translations to: {OUTPUT_EXCEL}")
display(out_df.head(20))