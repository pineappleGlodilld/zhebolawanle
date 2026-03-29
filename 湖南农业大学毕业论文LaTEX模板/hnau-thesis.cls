\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hnau-thesis}[2026/03/10 HNAU thesis template]

\newif\ifhnau@arts
\DeclareOption{arts}{\hnau@artstrue}
\DeclareOption{science}{\hnau@artsfalse}
\ExecuteOptions{science}
\PassOptionsToClass{UTF8,a4paper,12pt,scheme=plain}{ctexart}
\ProcessOptions\relax
\LoadClass{ctexart}

\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{tocloft}
\RequirePackage{fancyhdr}
\RequirePackage{caption}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{etoolbox}
\RequirePackage{xparse}
\RequirePackage{keyval}
\RequirePackage[hidelinks]{hyperref}

\defaultfontfeatures{Ligatures=TeX,Scale=MatchLowercase}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{TeX Gyre Termes}}
\IfFontExistsTF{SimSun}{\setCJKmainfont[AutoFakeBold=2]{SimSun}}{\setCJKmainfont[AutoFakeBold=2]{FandolSong-Regular}}
\IfFontExistsTF{SimHei}{\setCJKsansfont[AutoFakeBold=2]{SimHei}}{\setCJKsansfont{FandolHei-Regular}}
\IfFontExistsTF{KaiTi}{\setCJKmonofont{KaiTi}}{\setCJKmonofont{FandolKai-Regular}}

\IfFontExistsTF{STXingkai}{\newCJKfontfamily\hnauxingkai[AutoFakeBold=2]{STXingkai}}{%
  \IfFontExistsTF{华文行楷}{\newCJKfontfamily\hnauxingkai{华文行楷}}{%
    \IfFontExistsTF{KaiTi}{\newCJKfontfamily\hnauxingkai{KaiTi}}{\newCJKfontfamily\hnauxingkai{FandolKai-Regular}}%
  }%
}
\IfFontExistsTF{SimHei}{\newCJKfontfamily\hnauheiti[AutoFakeBold=2]{SimHei}}{\newCJKfontfamily\hnauheiti{FandolHei-Regular}}
\IfFontExistsTF{KaiTi}{\newCJKfontfamily\hnaukaiti{KaiTi}}{\newCJKfontfamily\hnaukaiti{FandolKai-Regular}}
\IfFontExistsTF{Times New Roman}{\newfontfamily\hnautimes{Times New Roman}}{\newfontfamily\hnautimes{TeX Gyre Termes}}

\newcommand{\HNAU@covergeometry}{%
  \newgeometry{top=2.54cm,bottom=2.54cm,left=2.6cm,right=2.6cm,headheight=0pt,headsep=0pt,footskip=1.75cm}%
}
\newcommand{\HNAU@bodygeometry}{%
  \newgeometry{top=2.54cm,bottom=2.54cm,left=2.6cm,right=2.6cm,headheight=0pt,headsep=0pt,footskip=1.75cm}%
}
\newcommand{\HNAU@bodyfont}{\songti\fontsize{12pt}{22pt}\selectfont}
\newcommand{\HNAU@tocfont}{\songti\fontsize{12pt}{18pt}\selectfont}
\newcommand{\HNAU@fivefont}{\songti\fontsize{10.5pt}{18pt}\selectfont}
\newcommand{\HNAU@fiveenfont}{\hnautimes\fontsize{10.5pt}{18pt}\selectfont}

\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
\captionsetup{font={small},labelsep=space}

\fancypagestyle{hnaubody}{%
  \fancyhf{}
  \fancyfoot[C]{\songti\fontsize{10.5pt}{10.5pt}\selectfont\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

\renewcommand{\contentsname}{目\hspace{2em}录}
\renewcommand{\cfttoctitlefont}{\hfill\heiti\bfseries\zihao{2}}
\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftsecfont}{\songti\zihao{-4}}
\renewcommand{\cftsubsecfont}{\songti\zihao{-4}}
\renewcommand{\cftsubsubsecfont}{\songti\zihao{-4}}
\renewcommand{\cftsecpagefont}{\songti\zihao{-4}}
\renewcommand{\cftsubsecpagefont}{\songti\zihao{-4}}
\renewcommand{\cftsubsubsecpagefont}{\songti\zihao{-4}}
\setlength{\cftbeforetoctitleskip}{0pt}
\setlength{\cftaftertoctitleskip}{2\baselineskip}
\setlength{\cftbeforesecskip}{0pt}
\setlength{\cftsecindent}{0pt}
\setlength{\cftsubsecindent}{2em}
\setlength{\cftsubsubsecindent}{4em}
\setlength{\cftsecnumwidth}{3em}
\setlength{\cftsubsecnumwidth}{4.5em}
\setlength{\cftsubsubsecnumwidth}{6em}

\ifhnau@arts
  \renewcommand{\thesection}{\chinese{section}、}
  \renewcommand{\thesubsection}{（\chinese{subsection}）}
  \renewcommand{\thesubsubsection}{\arabic{subsubsection}、}
  \titleformat{\section}{\raggedright\heiti\bfseries\zihao{-3}}{\thesection}{0pt}{}
  \titleformat{\subsection}{\raggedright\heiti\bfseries\zihao{4}}{\thesubsection}{0pt}{}
  \titleformat{\subsubsection}{\raggedright\heiti\bfseries\zihao{-4}}{\thesubsubsection}{0.5em}{}
  \titlespacing*{\section}{0pt}{22pt}{11pt}
  \titlespacing*{\subsection}{0pt}{22pt}{11pt}
  \titlespacing*{\subsubsection}{2em}{11pt}{0pt}
\else
  \renewcommand{\thesection}{\arabic{section}}
  \renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
  \renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}
  \titleformat{\section}{\raggedright\heiti\bfseries\zihao{-3}}{\thesection}{0.5em}{}
  \titleformat{\subsection}{\raggedright\heiti\bfseries\zihao{4}}{\thesubsection}{0.5em}{}
  \titleformat{\subsubsection}{\raggedright\heiti\bfseries\zihao{-4}}{\thesubsubsection}{0.5em}{}
  \titlespacing*{\section}{0pt}{22pt}{11pt}
  \titlespacing*{\subsection}{0pt}{22pt}{11pt}
  \titlespacing*{\subsubsection}{0pt}{11pt}{0pt}
\fi

\makeatletter
\def\hnau@ctitle{毕业论文（设计）中文题目}
\def\hnau@etitle{TITLE OF GRADUATION PAPER}
\def\hnau@author{张三}
\def\hnau@studentid{202600000000}
\def\hnau@gradeclass{2026级XX专业（X）班}
\def\hnau@advisor{李四}
\def\hnau@advisorwithtitle{李四\ 教授}
\def\hnau@college{XX学院}
\def\hnau@ecollege{College of XX}
\def\hnau@city{长沙}
\def\hnau@postcode{410128}
\def\hnau@submitdate{2026年3月}
\def\hnau@statementtext{本人郑重声明：所呈交的本科毕业论文（设计）是本人在指导老师的指导下，进行研究工作所取得的成果，成果不存在知识产权争议。除文中已经注明引用的内容外，本论文不含任何其他个人或集体已经发表或撰写过的作品成果。对本文的研究做出重要贡献的个人和集体在文中均作了明确的说明并表示了谢意。本人完全意识到本声明的法律结果由本人承担。}

\define@key{hnau}{ctitle}{\def\hnau@ctitle{#1}}
\define@key{hnau}{etitle}{\def\hnau@etitle{#1}}
\define@key{hnau}{author}{\def\hnau@author{#1}}
\define@key{hnau}{studentid}{\def\hnau@studentid{#1}}
\define@key{hnau}{gradeclass}{\def\hnau@gradeclass{#1}}
\define@key{hnau}{advisor}{\def\hnau@advisor{#1}}
\define@key{hnau}{advisorwithtitle}{\def\hnau@advisorwithtitle{#1}}
\define@key{hnau}{college}{\def\hnau@college{#1}}
\define@key{hnau}{ecollege}{\def\hnau@ecollege{#1}}
\define@key{hnau}{city}{\def\hnau@city{#1}}
\define@key{hnau}{postcode}{\def\hnau@postcode{#1}}
\define@key{hnau}{submitdate}{\def\hnau@submitdate{#1}}
\define@key{hnau}{statementtext}{\def\hnau@statementtext{#1}}
\newcommand{\HNAUsetup}[1]{\setkeys{hnau}{#1}}

\renewcommand{\refname}{参考文献}
\patchcmd{\thebibliography}{\section*{\refname}}{\clearpage\section*{\refname}\phantomsection\addcontentsline{toc}{section}{\refname}}{}{}
\AtBeginEnvironment{thebibliography}{\HNAU@fivefont\setstretch{1.5}}

\AtBeginDocument{\HNAU@bodygeometry\HNAU@bodyfont\pagestyle{empty}}

\newcommand{\makecover}{%
  \clearpage
  \HNAU@covergeometry
  \pagestyle{empty}
  \thispagestyle{empty}
  \begin{center}
    \setstretch{1}
    \vspace*{0.8\baselineskip}
    {\hnauxingkai\bfseries\fontsize{26bp}{32bp}\selectfont 湖\ 南\ 农\ 业\ 大\ 学\par}
    \vspace{1.2\baselineskip}
    {\hnauheiti\bfseries\fontsize{26bp}{32bp}\selectfont 全日制普通本科生毕业论文（设计）\par}
    \vspace{4\baselineskip}
    {\hnauheiti\bfseries\fontsize{18bp}{26bp}\selectfont \hnau@ctitle\par}
    \vspace{1.5\baselineskip}
    {\hnautimes\bfseries\fontsize{14bp}{20bp}\selectfont \MakeUppercase{\hnau@etitle}\par}
    \vfill
    \renewcommand{\arraystretch}{1.9}
    \begin{tabularx}{0.88\textwidth}{>{\raggedleft\arraybackslash}p{10em}>{\centering\arraybackslash}X}
      {\heiti\bfseries\zihao{-3} 学生姓名：} & {\hnaukaiti\bfseries\zihao{-3} \hnau@author}\\
      {\heiti\bfseries\zihao{-3} 学\hspace{2em}号：} & {\hnaukaiti\bfseries\zihao{-3} \hnau@studentid}\\
      {\heiti\bfseries\zihao{-3} 年级专业及班级：} & {\hnaukaiti\bfseries\zihao{-3} \hnau@gradeclass}\\
      {\heiti\bfseries\zihao{-3} 指导老师及职称：} & {\hnaukaiti\bfseries\zihao{-3} \ifstrempty{\hnau@advisorwithtitle}{\hnau@advisor}{\hnau@advisorwithtitle}}\\
      {\heiti\bfseries\zihao{-3} 学\hspace{2em}院：} & {\hnaukaiti\bfseries\zihao{-3} \hnau@college}\\
    \end{tabularx}
    \vfill
    {\heiti\zihao{-3} 湖南·\hnau@city\par}
    \vspace{1.2\baselineskip}
    {\heiti\zihao{-3} 提交日期：\hnau@submitdate\par}
  \end{center}
  \clearpage
  \HNAU@bodygeometry
  \HNAU@bodyfont
}

\newcommand{\makestatement}{%
  \clearpage
  \HNAU@covergeometry
  \pagestyle{empty}
  \thispagestyle{empty}
  \begin{center}
    {\heiti\zihao{3} 湖南农业大学全日制普通本科生毕业论文（设计）\par}
    \vspace{2\baselineskip}
    {\heiti\bfseries\zihao{2} 诚\hspace{2em}信\hspace{2em}声\hspace{2em}明\par}
  \end{center}
  \vspace{3\baselineskip}
  {\songti\zihao{-4}\setstretch{1.6}\noindent \hnau@statementtext\par}
  \vfill
  \begin{flushright}
    {\songti\zihao{-4} 毕业论文（设计）作者签名：\makebox[7cm][l]{\hnau@author}\par}
    \vspace{2\baselineskip}
    {\songti\zihao{-4} 年\hspace{3em}月\hspace{3em}日\par}
  \end{flushright}
  \clearpage
  \HNAU@bodygeometry
  \HNAU@bodyfont
}

\newcommand{\maketoc}{%
  \clearpage
  \HNAU@bodygeometry
  \pagestyle{empty}
  \thispagestyle{empty}
  {\HNAU@tocfont\setstretch{1.5}\tableofcontents}
  \clearpage
  \HNAU@bodyfont
}

\NewDocumentCommand{\makeabstracts}{m m m m}{%
  \clearpage
  \HNAU@bodygeometry
  \pagenumbering{arabic}
  \setcounter{page}{1}
  \pagestyle{hnaubody}
  \thispagestyle{hnaubody}
  \begin{center}
    {\heiti\bfseries\zihao{-2} \hnau@ctitle\par}
    \vspace{2\baselineskip}
    {\songti\zihao{5} 学\hspace{2em}生：\hnau@author\par}
    {\songti\zihao{5} 指导老师：\hnau@advisor\par}
    {\songti\zihao{5} （湖南农业大学\hnau@college，\hnau@city\ \hnau@postcode）\par}
  \end{center}
  \vspace{\baselineskip}
  \phantomsection\addcontentsline{toc}{section}{摘要}
  {\noindent{\heiti\bfseries\zihao{-4} 摘要：}{\HNAU@fivefont #1}\par}
  \phantomsection\addcontentsline{toc}{section}{关键词}
  {\noindent{\heiti\bfseries\zihao{-4} 关键词：}{\HNAU@fivefont #2}\par}
  \vspace{\baselineskip}
  \begin{center}
    {\hnautimes\bfseries\fontsize{14bp}{18bp}\selectfont \hnau@etitle\par}
    \vspace{\baselineskip}
    {\HNAU@fiveenfont Student: \hnau@author\par}
    {\HNAU@fiveenfont Tutor: \hnau@advisor\par}
    {\HNAU@fiveenfont (\hnau@ecollege, Hunan Agricultural University, \hnau@city\ \hnau@postcode, China)\par}
  \end{center}
  \vspace{\baselineskip}
  {\noindent{\hnautimes\bfseries\fontsize{12pt}{18pt}\selectfont Abstract: }{\HNAU@fiveenfont #3}\par}
  {\noindent{\hnautimes\bfseries\fontsize{12pt}{18pt}\selectfont Key words: }{\HNAU@fiveenfont #4}\par}
  \vspace{\baselineskip}
  \HNAU@bodyfont
}

\NewDocumentCommand{\acknowledgements}{m}{%
  \clearpage
  \phantomsection
  \addcontentsline{toc}{section}{致谢}
  {\centering\heiti\bfseries\zihao{-3} 致\hspace{2em}谢\par}
  \vspace{\baselineskip}
  {\songti\zihao{-4} #1\par}
}

\NewDocumentCommand{\appendixpage}{m +m}{%
  \clearpage
  \phantomsection
  \addcontentsline{toc}{section}{附录}
  {\raggedright\heiti\bfseries\zihao{-3} 附录\par}
  \vspace{\baselineskip}
  {\raggedright\heiti\bfseries\zihao{-4} #1\par}
  \vspace{0.5\baselineskip}
  {\songti\zihao{-4} #2\par}
}
\makeatother