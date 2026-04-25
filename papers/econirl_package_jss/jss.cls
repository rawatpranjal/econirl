%% Minimal local stub for jss.cls.
%% Defines the JSS-specific commands as article-equivalents so the paper
%% compiles for verification. Replace this file with the official jss.cls
%% (available from JSS once a manuscript is submitted) before final
%% submission.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{jss}[2026/04/25 Local stub for JSS submission template]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt]{article}

\RequirePackage[left=1in,right=1in,top=1in,bottom=1in]{geometry}
\RequirePackage{amsmath,amssymb}
\RequirePackage{hyperref}
\hypersetup{colorlinks,citecolor=blue,filecolor=blue,linkcolor=blue,urlcolor=blue}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{natbib}
\RequirePackage{url}

\bibliographystyle{plainnat}

%% JSS-specific metadata commands (no-ops in the stub; record values
%% for the abstract block to render at \begin{document}).
\newcommand{\Plaintitle}[1]{}
\newcommand{\Plainauthor}[1]{}
\newcommand{\Shorttitle}[1]{}
\newcommand{\Plainabstract}[1]{}
\newcommand{\Plainkeywords}[1]{}
\newcommand{\Address}[1]{}

\def\@jsskeywords{}
\newcommand{\Keywords}[1]{\gdef\@jsskeywords{#1}}

\def\@jssabstract{}
\newcommand{\Abstract}[1]{\gdef\@jssabstract{#1}}

\AtBeginDocument{%
  \maketitle
  \ifx\@jssabstract\@empty\else
    \begin{quote}\small\textbf{Abstract.} \@jssabstract\end{quote}%
  \fi
  \ifx\@jsskeywords\@empty\else
    \par\noindent\textit{Keywords:} \@jsskeywords\par\bigskip
  \fi
}

%% JSS authorship combinators. Article's \and uses a tabular template
%% that conflicts with our centred maketitle, so override unconditionally.
\def\And{\quad\textbullet\quad}
\def\AND{\par}
\def\and{\quad\textbullet\quad}

%% Override \maketitle to avoid the article-class tabular template that
%% conflicts with the JSS \And / \AND combinators in the author block.
\renewcommand{\maketitle}{%
  \begingroup
  \par\centering
  {\LARGE\bfseries\@title\par}\bigskip
  {\large\@author\par}\medskip
  {\@date\par}
  \endgroup
  \par\bigskip
}

%% JSS inline code and package commands.
\providecommand{\pkg}[1]{\textbf{#1}}
\providecommand{\proglang}[1]{\textsf{#1}}
\providecommand{\code}[1]{\texttt{#1}}
\providecommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}
