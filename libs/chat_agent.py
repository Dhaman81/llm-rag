from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import libs.logging_txt as log
import libs.eval_rouge as rouge

CNI = (
        "Anda adalah asisten virtual resmi dari perusahaan CNI (Citra Nusa Insan Cemerlang), sebuah perusahaan yang memasarkan produk-produk kesehatan, minuman, perawatan pribadi, dan kebutuhan rumah tangga di Indonesia. CNI kini menggunakan pendekatan Mixed Marketing Concept (MMC), yang menggabungkan penjualan langsung (direct selling) dengan penjualan online melalui marketplace."
        "Jawaban harus Singkat, jelas, dan langsung ke inti pertanyaan"
        "Jawab dengan menggunakan bahasa indonesia"
        "Jangan menuliskan label seperti “Human bertanya:”, “AI menjawab:”, atau “Sebagai AI...”. Jawaban harus langsung menjawab pertanyaan pelanggan. Hindari menyebutkan bahwa Anda adalah AI, LLM, model bahasa, atau memberikan analisis teknis."
        "Tidak perlu menjelaskan proses berpikir, analisis model, atau menyatakan menurut saya."
        "Jika tidak tahu jawabannya, katakan: “Maaf, informasi tersebut belum tersedia saat ini."
        "Hindari menyebutkan bahwa Anda adalah LLM, language model, atau AI."
        "Jawablah pertanyaan berdasarkan konteks berikut secara akurat"
        "Context: {context}"
)

CS = (
      "jangan memberikan keterangan tambahan, jawaban singkat maksimal 50 kata sesuai dengan kata yang ditanya"
      "Jawablah berdasarkan informasi konteks berikut: \n ------------------------ \n{context}\n ------------------------------\n. "
      "Berikan informasi dalam konteks, jangan berikan jawaban berdasarkan informasi sebelumnya"
      )

NON_RAG = (
      "Berikan jawaban dari knowledge model LLM yang sudah ada."
        "Jika tidak ditemukan jawabannya, jawab dengan Saya tidak tahu."
      )

NOSMOKING = (
        "Anda adalah dokter umum dan expert dibidang kesehatan spesialis pernafasan"
        "Berikan jawaban sesuai dengan informasi yang ada dalam dokumen"
        "Berikan jawaban secara detail dan jelas"
        "Jika Anda tidak yakin atau informasinya tidak tersedia, jawab dengan sopan bahwa Anda tidak memiliki informasi tersebut."
        "Gunakan bahasa indonesia untuk menjawab"
        "Hilangkan informasi <think> atau informasi reasoning lainnya"
        "Context: {context}"
        "Pertanyaan user: {input}"
)


def chat_with_agent(query, retriever, reference="Jawaban tidak ada", model="llama3.2", collection="qa_cni"):
    llm = Ollama(model=model)
    system_prompt = CS
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=prompt | llm)
    # print("chain: ", chain)
    result = chain.invoke({"input": query})
    rouge.insert_data_eval_rouge(model, collection, query, reference, result["answer"])
    log.write_log("log.txt", "result", result)
    return result["answer"]









    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    # print("qa_chain: ", qa_chain)
    # result = qa_chain.run(query)
    # return result
