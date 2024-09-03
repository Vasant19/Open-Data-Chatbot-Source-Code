import { NextResponse } from 'next/server';
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { createSqlQueryChain } from "langchain/chains/sql_db";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";

export async function POST(request) {
  try {
    const { query: userQuestion, model } = await request.json();

    if (!process.env.DB_HOST || !process.env.DB_USER || !process.env.DB_PASSWORD || !process.env.DB_NAME) {
      throw new Error("Database environment variables are not set.");
    }

    const datasource = new DataSource({
      type: "mysql",
      host: process.env.DB_HOST,
      port: 3306,
      username: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
    });

    await datasource.initialize();

    const db = await SqlDatabase.fromDataSourceParams({
      appDataSource: datasource,
    });

    const llm = new ChatOpenAI({ modelName: model, temperature: 0 });
    const chain = await createSqlQueryChain({
      llm,
      db,
      dialect: "mysql",
    });

    const SYSTEM_PROMPT = `Double check the user's {dialect} query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

    Output the final SQL query only.`;

    const prompt = await ChatPromptTemplate.fromMessages([
      ["system", SYSTEM_PROMPT],
      ["human", "{query}"],
    ]).partial({ dialect: "mysql" });

    const validationChain = prompt.pipe(llm).pipe(new StringOutputParser());

    const fullChain = RunnableSequence.from([
      {
        query: async (i) => chain.invoke(i),
      },
      validationChain,
    ]);

    const sqlQuery = await fullChain.invoke({
        question: userQuestion,
      });
      const queryResult = await db.run(sqlQuery);
      console.log('Raw queryResult:', JSON.stringify(queryResult, null, 2));
  
      await datasource.destroy();
  
      // Extract the raw value
      let formattedResult = '';
      if (typeof queryResult === 'string') {
        try {
          const parsedResult = JSON.parse(queryResult);
          if (Array.isArray(parsedResult) && parsedResult.length > 0) {
            formattedResult = Object.values(parsedResult[0])[0];
          }
        } catch (e) {
          console.error('Error parsing queryResult:', e);
          formattedResult = queryResult;
        }
      } else if (Array.isArray(queryResult) && queryResult.length > 0) {
        formattedResult = Object.values(queryResult[0])[0];
      } else {
        formattedResult = JSON.stringify(queryResult);
      }
  
      console.log('Final formattedResult:', formattedResult);
  
      return NextResponse.json({ sqlQuery, queryResult: formattedResult });
    } catch (error) {
      console.error('Error:', error);
      return NextResponse.json({ error: 'An error occurred while processing your request.' }, { status: 500 });
    }
  }