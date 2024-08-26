#### Testing AopenAI CLI
```
export OPENAI_API_KEY=your-key
openai api chat.completions.create -m gpt-3.5-turbo -g "user" "Say hello, OpenAI CLI!"
```

#### Monitor Fine-tuning
```
openai api fine_tunes.follow -i <fine_tune_id>

```