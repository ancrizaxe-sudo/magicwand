import { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, Loader2 } from "lucide-react";
import { api } from "../lib/api";

export default function Docs() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const [list, setList] = useState(null);
  const [doc, setDoc] = useState(null);

  useEffect(() => { api.get("/docs").then((r) => setList(r.data.docs)).catch(() => setList([])); }, []);
  useEffect(() => {
    if (!list) return;
    const target = slug || list[0]?.slug;
    if (!target) return;
    api.get(`/docs/${target}`).then((r) => setDoc(r.data)).catch(() => setDoc(null));
  }, [slug, list]);

  if (!list) return <Center msg={<><Loader2 className="animate-spin inline mr-2"/>Loading …</>}/>;

  return (
    <div className="max-w-7xl mx-auto px-6 py-12 grid lg:grid-cols-12 gap-8">
      <aside className="lg:col-span-3 reveal">
        <div className="text-xs uppercase tracking-[0.22em] text-[var(--leaf-700)] mb-2">docs</div>
        <h2 className="text-2xl mb-4" style={{fontFamily:"var(--font-display)"}}>AVLE-C handbook</h2>
        <nav className="flex lg:flex-col gap-2">
          {list.map((d) => (
            <Link key={d.slug} to={`/docs/${d.slug}`} data-testid={`doc-link-${d.slug}`}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition ${
                    (slug || list[0]?.slug) === d.slug
                      ? "bg-[var(--leaf-700)] text-white"
                      : "text-[var(--leaf-900)] hover:bg-[var(--leaf-50)]"
                  }`}>
              <BookOpen size={14}/> {d.title}
            </Link>
          ))}
        </nav>
      </aside>

      <article className="lg:col-span-9 leaf-card rounded-2xl p-8 reveal reveal-d1">
        {!doc ? (
          <div className="text-[var(--stone)]"><Loader2 className="animate-spin inline mr-2"/>Loading doc …</div>
        ) : (
          <div data-testid="doc-article" className="prose prose-stone max-w-none prose-headings:font-display prose-headings:tracking-tight prose-h1:text-4xl prose-h2:text-2xl prose-h3:text-xl prose-a:text-[var(--leaf-700)] prose-strong:text-[var(--leaf-900)] prose-code:text-[var(--leaf-800)] prose-code:bg-[var(--leaf-50)] prose-code:px-1 prose-code:rounded">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{doc.markdown}</ReactMarkdown>
          </div>
        )}
      </article>
    </div>
  );
}

const Center = ({ msg }) => (
  <div className="max-w-4xl mx-auto px-6 py-24 text-center text-[var(--stone)]">{msg}</div>
);
