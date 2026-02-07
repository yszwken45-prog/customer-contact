"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain import SerpAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
import utils
import constants as ct



############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # Agent Executorを作成
    initialize_agent_executor()


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        # 会話履歴の合計トークン数を加算する用の変数
        st.session_state.total_tokens = 0

        # フィードバックボタンで「はい」を押下した後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_yes_flg = False
        # フィードバックボタンで「いいえ」を押下した後に入力エリアを表示するためのフラグ
        st.session_state.feedback_no_flg = False
        # LLMによる回答生成後、フィードバックボタンを表示するためのフラグ
        st.session_state.answer_flg = False
        # フィードバックボタンで「いいえ」を押下後、フィードバックを送信するための入力エリアからの入力を受け付ける変数
        st.session_state.dissatisfied_reason = ""
        # フィードバック送信後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_no_reason_send_flg = False


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


# def initialize_logger():
#     """
#     ログ出力の設定
#     """
#     os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)

#     logger = logging.getLogger(ct.LOGGER_NAME)

#     if logger.hasHandlers():
#         return

#     log_handler = TimedRotatingFileHandler(
#         os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
#         when="D",
#         encoding="utf8"
#     )
#     formatter = logging.Formatter(
#         f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
#     )
#     log_handler.setFormatter(formatter)
#     logger.setLevel(logging.INFO)
#     logger.addHandler(log_handler)


def initialize_logger():
    # 1. フォルダ作成（失敗しても無視して進む）
    try:
        os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    except:
        pass

    logger = logging.getLogger(ct.LOGGER_NAME)
    if logger.handlers:
        logger.handlers.clear()

    # 2. 標準出力（コンソール）用の設定を追加
    # これにより、Streamlit Cloudの「Logs」画面にログが出るようになります
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s session_id={st.session_state.get('session_id', 'N/A')}: %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 3. ファイル出力の設定（念のため残す）
    try:
        log_handler = TimedRotatingFileHandler(
            os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
            when="D",
            encoding="utf8"
        )
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    except Exception as e:
        # クラウド環境などでファイルが作れなくても、ここでのエラーは無視する
        logger.warning(f"File log failed: {e}")

    logger.setLevel(logging.INFO)
    logger.info("Logger initialized (Stream + File)")

def initialize_agent_executor():
    """
    画面読み込み時にAgent Executor（AIエージェント機能の実行を担当するオブジェクト）を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにAgent Executorが作成済みの場合、後続の処理を中断
    if "agent_executor" in st.session_state:
        return
    
    # 消費トークン数カウント用のオブジェクトを用意
    st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
    
    st.session_state.llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE, streaming=True)

    # 各Tool用のChainを作成
    st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
    st.session_state.service_doc_chain = utils.create_rag_chain(ct.DB_SERVICE_PATH)
    st.session_state.company_doc_chain = utils.create_rag_chain(ct.DB_COMPANY_PATH)
    st.session_state.rag_chain = utils.create_rag_chain(ct.DB_ALL_PATH)

    # 新規追加：それぞれのDBパスから専用のChainを作成
    st.session_state.design_tech_doc_chain = utils.create_rag_chain(ct.DB_DESIGN_TECH_PATH)
    st.session_state.compliance_doc_chain = utils.create_rag_chain(ct.DB_COMPLIANCE_PATH)
    st.session_state.logistics_doc_chain = utils.create_rag_chain(ct.DB_LOGISTICS_PATH)

    # Web検索用のToolを設定するためのオブジェクトを用意
    search = SerpAPIWrapper()
    # Agent Executorに渡すTool一覧を用意
    tools = [
        # 会社に関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,
            func=utils.run_company_doc_chain,
            description=ct.SEARCH_COMPANY_INFO_TOOL_DESCRIPTION
        ),
        # サービスに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,
            func=utils.run_service_doc_chain,
            description=ct.SEARCH_SERVICE_INFO_TOOL_DESCRIPTION
        ),
        # 顧客とのやり取りに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME,
            func=utils.run_customer_doc_chain,
            description=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION
        ),
        # Web検索用のTool
        Tool(
            name = ct.SEARCH_WEB_INFO_TOOL_NAME,
            func=search.run,
            description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION
        ),
        # 1. デザイン・制作技術リファレンス用のTool
        Tool(
            name="search_design_technical_tool",
            func=utils.run_design_tech_doc_chain, # 対応するchainを別途定義
            description="デザインの入稿規定（解像度・形式）、印刷技術の仕様、商品の素材詳細や寸法など、制作・技術に関する詳細情報を参照したい時に使う"
        ),
        # 2. 規約・ガバナンス参照用のTool
        Tool(
            name="search_compliance_policy_tool",
            func=utils.run_compliance_doc_chain, # 対応するchainを別途定義
            description="利用規約、プライバシーポリシー、環境認証（エシカル）の基準、株主優待の権利確定条件など、法的・公式なルールを確認したい時に使う"
        ),
        # 3. 物流・代行出荷オペレーション用のTool
        Tool(
            name="search_logistics_operation_tool",
            func=utils.run_logistics_doc_chain, # 対応するchainを別途定義
            description="代行出荷の具体的なフロー、梱包仕様、配送リードタイム、配送料金など、物流実務に関する情報を参照したい時に使う"
        )
    ]

    # Agent Executorの作成
    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )