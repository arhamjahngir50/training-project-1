"""
Production RAG Test Corpus
Multi-domain, rich enough to stress-test all retrieval strategies.
No LLM needed — pure retrieval evaluation.
"""

DOCUMENTS = [
    # ── Medical ──────────────────────────────────────────────────────────
    {
        "id": "med_001", "domain": "medical",
        "title": "Type 2 Diabetes Treatment",
        "text": (
            "Type 2 diabetes is managed through a combination of lifestyle modifications and pharmacological interventions. "
            "First-line treatment is metformin, which reduces hepatic glucose production and improves insulin sensitivity. "
            "When metformin is insufficient, GLP-1 receptor agonists such as semaglutide or liraglutide are added. "
            "SGLT-2 inhibitors like empagliflozin offer cardiovascular and renal protection in high-risk patients. "
            "Blood glucose targets: HbA1c below 7% for most adults, 7.5–8% for elderly or those with hypoglycemia risk. "
            "Diet intervention: low glycaemic index foods, reducing refined carbohydrates, 150 minutes of aerobic exercise weekly. "
            "Bariatric surgery is considered when BMI exceeds 35 with uncontrolled diabetes despite optimal medical therapy."
        )
    },
    {
        "id": "med_002", "domain": "medical",
        "title": "Hypertension Management",
        "text": (
            "Hypertension is defined as sustained blood pressure above 130/80 mmHg. "
            "ACE inhibitors or ARBs are preferred first-line agents, especially in diabetics and those with chronic kidney disease. "
            "Calcium channel blockers such as amlodipine are effective in older patients and those of African descent. "
            "Thiazide diuretics reduce fluid volume and are cost-effective options. "
            "Lifestyle modifications include the DASH diet, sodium restriction below 2.3 g/day, weight loss, and regular aerobic activity. "
            "Resistant hypertension requires adding spironolactone as a fourth agent. "
            "White-coat hypertension should be confirmed with ambulatory 24-hour blood pressure monitoring before starting treatment."
        )
    },
    {
        "id": "med_003", "domain": "medical",
        "title": "COVID-19 Pathophysiology",
        "text": (
            "SARS-CoV-2 enters cells via the ACE2 receptor, which is highly expressed in lung alveolar cells, cardiac tissue, and kidneys. "
            "The spike protein binds ACE2 with high affinity; TMPRSS2 primes the spike for cell entry. "
            "Severe disease is characterised by a cytokine storm: excessive IL-6, TNF-alpha, and interferon-gamma release. "
            "ARDS develops in 5–10% of hospitalised patients due to alveolar damage and impaired gas exchange. "
            "Dexamethasone reduces 28-day mortality in patients requiring oxygen or mechanical ventilation. "
            "Long COVID affects 10–30% of infected individuals, with symptoms including fatigue, cognitive impairment, and dyspnoea persisting beyond 12 weeks."
        )
    },

    # ── Finance ───────────────────────────────────────────────────────────
    {
        "id": "fin_001", "domain": "finance",
        "title": "2008 Financial Crisis Causes",
        "text": (
            "The 2008 global financial crisis originated in the US subprime mortgage market. "
            "Banks issued mortgages to borrowers with poor credit ratings, bundled them into mortgage-backed securities (MBS), and sold tranches globally. "
            "Credit rating agencies assigned AAA ratings to toxic instruments due to flawed models and conflicts of interest. "
            "When US house prices declined 30% from 2006 peaks, default rates surged, MBS values collapsed, and interbank lending froze. "
            "Lehman Brothers filed for bankruptcy in September 2008 with $613 billion in debt. "
            "The Federal Reserve and US Treasury deployed $700 billion TARP bailout and unprecedented quantitative easing. "
            "Global GDP contracted 2.1% in 2009; unemployment in the US reached 10%. "
            "Dodd-Frank Act 2010 introduced stress testing, Volcker Rule, and systemic risk oversight to prevent recurrence."
        )
    },
    {
        "id": "fin_002", "domain": "finance",
        "title": "Keynesian vs Austrian Economics on Inflation",
        "text": (
            "Keynesian economics views inflation as demand-pull: when aggregate demand exceeds productive capacity, prices rise. "
            "Keynesians favour counter-cyclical fiscal policy — increasing government spending in recessions to stimulate demand. "
            "The Phillips Curve in Keynesian theory suggests a short-run trade-off between inflation and unemployment. "
            "Austrian economics, pioneered by von Mises and Hayek, attributes inflation primarily to monetary expansion. "
            "Austrians argue that central bank credit expansion distorts the capital structure, creating artificial booms followed by inevitable busts. "
            "The Austrian Business Cycle Theory holds that low interest rates cause malinvestment in long-duration capital goods. "
            "Keynesians accept short-run price stickiness; Austrians emphasise long-run monetary neutrality and price discovery. "
            "Both schools agree hyperinflation is catastrophic, but disagree fundamentally on the correct policy response to recessions."
        )
    },
    {
        "id": "fin_003", "domain": "finance",
        "title": "Venture Capital and Startup Funding",
        "text": (
            "Venture capital funds raise committed capital from limited partners — pension funds, endowments, family offices. "
            "A typical fund lifecycle is 10 years: 3–4 years investing, remainder managing and exiting portfolio companies. "
            "Pre-seed rounds are typically $250K–$1M from angels or pre-seed funds for idea validation. "
            "Series A funding ranges from $5M–$15M and requires demonstrated product-market fit and revenue traction. "
            "Valuation methods include revenue multiples (10–20x ARR for SaaS), discounted cash flow, and comparable transactions. "
            "Liquidation preferences ensure VCs recoup investment before founders in downside exits. "
            "The power law governs VC returns: top 10% of investments generate 90% of fund returns. "
            "Exit routes include IPO, acquisition, secondary sale, or recapitalisation."
        )
    },

    # ── Technology ────────────────────────────────────────────────────────
    {
        "id": "tech_001", "domain": "technology",
        "title": "Transformer Architecture",
        "text": (
            "The Transformer architecture, introduced in 'Attention is All You Need' (Vaswani et al., 2017), replaced recurrent networks with self-attention. "
            "Self-attention computes query, key, and value matrices from input embeddings. "
            "Attention scores are computed as softmax(QK^T / sqrt(d_k)) * V, allowing each token to attend to all others. "
            "Multi-head attention runs several attention heads in parallel, capturing different relationship types. "
            "Positional encodings inject sequence order since attention is permutation-invariant. "
            "The encoder-decoder architecture: encoder maps input to contextual representations; decoder generates output autoregressively. "
            "BERT uses encoder-only for understanding tasks; GPT uses decoder-only for generation; T5 uses full encoder-decoder. "
            "Scaling laws show that model performance improves predictably with compute, data, and parameter count."
        )
    },
    {
        "id": "tech_002", "domain": "technology",
        "title": "Kubernetes Container Orchestration",
        "text": (
            "Kubernetes (K8s) is an open-source container orchestration platform developed by Google and donated to CNCF. "
            "Core primitives: Pod (smallest deployable unit, one or more containers), Deployment (manages replica sets), Service (stable networking endpoint), ConfigMap and Secret (configuration injection). "
            "The control plane consists of: kube-apiserver (REST gateway), etcd (distributed key-value store for cluster state), kube-scheduler (places pods on nodes), kube-controller-manager (reconciliation loops). "
            "Worker nodes run: kubelet (node agent), kube-proxy (iptables rules for service routing), container runtime (containerd or CRI-O). "
            "Horizontal Pod Autoscaler scales deployments based on CPU, memory, or custom metrics. "
            "StatefulSets provide stable network identity and persistent storage for databases. "
            "Helm charts package applications as versioned, configurable release bundles."
        )
    },
    {
        "id": "tech_003", "domain": "technology",
        "title": "Retrieval Augmented Generation",
        "text": (
            "Retrieval Augmented Generation (RAG) combines a retrieval system with a generative language model to answer questions grounded in external knowledge. "
            "The indexing pipeline chunks documents into 256–512 token segments and encodes them with a dense embedding model. "
            "At query time, the question is embedded and the top-k most similar chunks are retrieved by approximate nearest neighbour search. "
            "Advanced RAG techniques include query rewriting, HyDE (Hypothetical Document Embeddings), and cross-encoder reranking. "
            "Hybrid retrieval combines dense vector search with sparse BM25 keyword search using Reciprocal Rank Fusion. "
            "Self-RAG trains the model to emit reflection tokens indicating whether retrieval is needed and whether retrieved content is relevant. "
            "Graph RAG builds a knowledge graph from documents to support multi-hop relational queries. "
            "Production RAG systems require evaluation metrics: faithfulness, answer relevance, context precision, and context recall."
        )
    },

    # ── History ────────────────────────────────────────────────────────────
    {
        "id": "hist_001", "domain": "history",
        "title": "World War II Pacific Theatre",
        "text": (
            "The Pacific War began with Japan's attack on Pearl Harbor on 7 December 1941, drawing the United States into World War II. "
            "Japan had already invaded Manchuria (1931) and China (1937) before expanding into Southeast Asia for oil and resources. "
            "The Battle of Midway in June 1942 was the turning point: the US destroyed four Japanese carriers, ending Japanese naval superiority. "
            "Island-hopping strategy, devised by Admiral Nimitz, bypassed heavily fortified Japanese positions to capture strategically vital islands. "
            "The atomic bombings of Hiroshima (6 August 1945) and Nagasaki (9 August 1945) killed 110,000–210,000 people. "
            "Japan surrendered on 15 August 1945 (V-J Day); the formal surrender was signed aboard USS Missouri on 2 September 1945. "
            "The Pacific War reshaped geopolitics: decolonisation accelerated, the Cold War began, and Japan adopted a pacifist constitution."
        )
    },
    {
        "id": "hist_002", "domain": "history",
        "title": "Industrial Revolution Economic Impact",
        "text": (
            "The Industrial Revolution, beginning in Britain circa 1760, transformed agrarian economies into industrial ones. "
            "Key innovations: steam engine (James Watt, 1769), spinning jenny (Hargreaves, 1764), power loom, and iron smelting with coke. "
            "Factory system replaced cottage industries, concentrating workers in urban manufacturing centres. "
            "Real wages in Britain rose 50–100% between 1760 and 1850, though distribution was highly unequal. "
            "Child labour, 14-hour workdays, and unsafe conditions defined early industrial working conditions. "
            "Railways cut freight costs by 80% and created national markets for goods. "
            "By 1850, Britain produced 50% of the world's iron and cotton cloth, and was the world's leading creditor nation. "
            "The revolution spread to Belgium, France, Germany, and the US by the mid-19th century."
        )
    },

    # ── Science ────────────────────────────────────────────────────────────
    {
        "id": "sci_001", "domain": "science",
        "title": "CRISPR Gene Editing",
        "text": (
            "CRISPR-Cas9 is a bacterial immune system adapted for precision gene editing, developed by Doudna and Charpentier (Nobel Prize 2020). "
            "The guide RNA (gRNA) is a ~20 nucleotide sequence complementary to the target DNA. "
            "Cas9 protein binds the gRNA, scans the genome, and creates a double-strand break at the target site. "
            "The cell repairs the break via NHEJ (error-prone, causing knockouts) or HDR (precise, using a donor template). "
            "Base editing and prime editing offer single-nucleotide precision without double-strand breaks, reducing off-target effects. "
            "Clinical applications: sickle cell disease cure (Casgevy approved by FDA Dec 2023), cancer immunotherapy via CAR-T modification, and potential hereditary blindness treatments. "
            "Ethical concerns include germline editing, equity of access, and the 2018 He Jiankui case involving edited human embryos."
        )
    },
    {
        "id": "sci_002", "domain": "science",
        "title": "Climate Change Mechanisms",
        "text": (
            "The greenhouse effect arises when atmospheric CO2, methane, and water vapour absorb and re-emit infrared radiation, warming the planet. "
            "Pre-industrial CO2 concentration was ~280 ppm; as of 2024 it exceeds 420 ppm, the highest in 3 million years. "
            "Feedback mechanisms: melting Arctic ice reduces albedo (reflectivity), amplifying warming; permafrost thaw releases stored methane. "
            "The IPCC AR6 report projects 1.5°C warming above pre-industrial by the early 2030s under current trajectories. "
            "Sea level is rising at 3.3 mm/year due to thermal expansion and glacial melt, threatening 1 billion coastal residents. "
            "Mitigation requires net-zero emissions by 2050: renewable energy, carbon capture, electrification of transport, and methane reduction. "
            "Adaptation measures include flood defences, drought-resistant crops, and managed coastal retreat."
        )
    },

    # ── Business / Operations ──────────────────────────────────────────────
    {
        "id": "biz_001", "domain": "business",
        "title": "Refund Policy for E-commerce",
        "text": (
            "Standard e-commerce refund policy: items must be returned within 30 days of delivery in original condition with tags attached. "
            "Orders over £50 qualify for free return shipping via prepaid label generated in the customer portal. "
            "Orders under £50 require the customer to pay return postage; the refund covers the item price only. "
            "Digital products and perishable goods are non-refundable. "
            "Refunds are processed within 5–7 business days to the original payment method. "
            "Damaged or defective items receive a full refund plus return shipping reimbursement regardless of order value. "
            "International orders over £50 are eligible for prepaid returns; duties and taxes are non-refundable. "
            "Customers can initiate returns via the portal, email support@store.com, or call the helpline Monday–Friday 9am–5pm GMT."
        )
    },
    {
        "id": "biz_002", "domain": "business",
        "title": "Product Pricing Strategy",
        "text": (
            "Pricing strategies include cost-plus, value-based, competitive, and dynamic pricing. "
            "Cost-plus adds a fixed margin (e.g., 40%) to production cost; reliable but ignores customer willingness to pay. "
            "Value-based pricing anchors price to perceived customer value — premium products often priced 3–5x cost. "
            "Penetration pricing sets low initial prices to capture market share, then raises them once adoption reaches critical mass. "
            "Price elasticity measures demand sensitivity: luxury goods are inelastic (–0.2 to –0.5), staples are elastic (–1 to –3). "
            "Psychological pricing: £9.99 instead of £10 exploits left-digit anchoring. "
            "Regional pricing adjusts for GDP per capita and competitive landscape — emerging markets often priced 40–60% below Western benchmarks."
        )
    },
]

# Flat list of just texts + ids for easy iteration
ALL_TEXTS = [(doc["id"], doc["title"], doc["text"]) for doc in DOCUMENTS]
ALL_DOMAINS = sorted(set(d["domain"] for d in DOCUMENTS))