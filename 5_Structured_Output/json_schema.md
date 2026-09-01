In your LangChain example, the schema is also an instruction contract for the model:
“Return these exact fields with these exact data types.” Your Python backend can
then safely send the result to React, where TypeScript can display it without guessing
how to parse a normal paragraph. JSON Schema is language-independent, so Python,
JavaScript, Java, and other systems can share the same data rules.

So a better statement is: “We use JSON because frontend and backend exchange data easily
in that format. We use JSON Schema to define and validate the exact structure of that
JSON.”
