2. conversation.item.create (History-Injection)
   Dieses Event fügt neue Items zum Conversation Context hinzu — also Messages, Function Calls, Function Call Responses. Es kann sowohl genutzt werden, um eine Konversationshistorie vorab zu befüllen, als auch um Items mitten in einem Stream hinzuzufügen. Core42
   Du kannst direkt nach dem Session-Start beliebig viele Items als user oder assistant einfügen — der Agent "glaubt" dann, diese Konversation habe wirklich stattgefunden:
   json{
   "type": "conversation.item.create",
   "item": {
   "type": "message",
   "role": "user",
   "content": [{ "type": "input_text", "text": "Ich bin Max, Nutzer-ID 42, Premium-Kunde." }]
   }
   }
   json{
   "type": "conversation.item.create",
   "item": {
   "type": "message",
   "role": "assistant",
   "content": [{ "type": "text", "text": "Verstanden, ich helfe dir gerne weiter, Max." }]
   }
   }
   Wichtig: Assistant Audio Messages können damit aktuell nicht befüllt werden — nur Text.
