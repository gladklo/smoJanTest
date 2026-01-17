# Vector Store Abfragen – Vorteile der verschiedenen Methoden

In **LangChain** bieten Vector Stores mehrere Möglichkeiten, Dokumente abzufragen. Dieses Dokument gibt eine strukturierte Übersicht über die **Vorteile der verschiedenen Abfragearten**, illustriert am Beispiel von Nike-Dokumenten.

---

## 1. `similarity_search` – String-basierte Suche

```python
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
```

### Vorteile

### Typische Einsatzszenarien

* Semantische Suche nach kurzen, direkten Antworten
* Arbeiten mit natürlichen Sprachfragen

---

## 2. `asimilarity_search` – Asynchrone String-Suche

```python
results = await vector_store.asimilarity_search(
    "When was Nike incorporated?"
)
```

### Vorteile

### Typische Einsatzszenarien

* Backends von Webanwendungen
* Echtzeit-Abfragen über viele Dokumente

---

## 3. `similarity_search_with_score` – Suche mit Ähnlichkeitswert

```python
results = vector_store.similarity_search_with_score(
    "What was Nike's revenue in 2023?"
)

doc, score = results[0]
print(score)
```

### Vorteile

### Typische Einsatzszenarien

* Vergleich mehrerer Dokumente
* Retrieval-Augmented Generation (RAG)
* Auswahl nur der relevantesten Chunks für Prompts

---

## 4. `similarity_search_by_vector` – Suche per Embedding

```python
embedding = embeddings.embed_query(
    "How were Nike's margins impacted in 2023?"
)

results = vector_store.similarity_search_by_vector(embedding)
```

### Vorteile

### Typische Einsatzszenarien

* Komplexe semantische Suche
* Multimodale Anwendungen (Text-, Bild-Embeddings)
* Präzise Steuerung von Ranking und Ähnlichkeit

---

## Zusammenfassung – Wann welche Methode?

| Methode                        | Vorteil / Use Case                               |
| ------------------------------ | ------------------------------------------------ |
| `similarity_search`            | Einfach, direkt, string-basiert                  |
| `asimilarity_search`           | Asynchron, skalierbar, parallelisierbar          |
| `similarity_search_with_score` | Ranking und Filterung nach Ähnlichkeit           |
| `similarity_search_by_vector`  | Volle Kontrolle, embedding-gesteuert, multimodal |

---

## Kernidee

* **String-basiert** → schnelle, einfache Suche
* **Score-basiert** → quantitative Ähnlichkeit
* **Vector-/Embedding-basiert** → maximale Kontrolle über semantische Ähnlichkeit
