/*
 * Copyright (C) Shihira Fung, 2016 <fengzhiping@hotmail.com>
 */

#ifndef CLTMPL_H_INCLUDED
#define CLTMPL_H_INCLUDED

#include <string>
#include <vector>
#include <memory>
#include <regex>
#include <stdexcept>

namespace gcl {

namespace cltmpl {

/*
 * symbol ::=
 *      '[a-zA-Z_][a-zA-Z_0-9]*'
 *
 * number_literal ::=
 *      '[-+]?[0-9]*\.?[0-9]*'
 *
 * string_literal ::=
 *      '"(\\.|[^"])*"'
 *
 * statement ::=
 *      'for' symbol ':' expression |
 *      'for' symbol ',' symbol ':' expression |
 *      'if' expression |
 *      'endfor' |
 *      'endif'
 *
 * unary_operator ::=
 *      '!'
 *
 * binary_operator ::=
 *      '||' | '&&' |
 *      '+' | '-' | '*' | '/' | '%' |
 *      '<' | '>' | '==' | '>=' | '<='
 *
 * expression ::=
 *      '(' expression ')' |
 *      symbol |
 *      string_literal |
 *      number_literal |
 *      unary_operator expression
 *      expression '[' expression ']' |
 *      expression binary_operator expression
 *
 * text_block ::=
 *      '(?:(?!\{[\{%]).)+'
 *
 * statement_block ::=
 *      '{{' statement '}}'
 *
 * expression_block ::=
 *      '{%' expression '%}'
 *
 * block ::=
 *      text_block |
 *      statement_block |
 *      expression_block
 *
 * template ::=
 *      { block }
 *
 */

class parse_error : public std::exception {
public:
    parse_error(const std::string& c,
            std::string::const_iterator i) :
            cause_(c), i_(i) {
        std::ostringstream os;
        os << c << " at " << &*i;
        what_arg_ = os.str();
    }

    const char* what() const noexcept override {
        return what_arg_.c_str();
    }

    const std::string& cause() {
        return cause_;
    }

    const std::string::const_iterator& pos() {
        return i_;
    }

protected:
    std::string cause_;
    std::string::const_iterator i_;
    std::string what_arg_;
};

class cl_block {
public:
    enum block_type {
        blk_type_expr,
        blk_type_text,
        blk_type_stat,
    };

protected:
    typedef std::string::const_iterator istr;

    istr src_beg_;
    istr src_end_;

    block_type blk_type_;

    cl_block(istr sb, istr se, block_type t) :
        src_beg_(sb), src_end_(se), blk_type_(t) { }
};

class value {
};

class expression_parser {
};

class expression_block : public cl_block {
public:
    expression_block(istr sb, istr se) :
        cl_block(sb, se, blk_type_expr) { }
};

class text_block : public cl_block {
public:
    text_block(istr sb, istr se) :
        cl_block(sb, se, blk_type_expr) { }
};

class statement_block : public cl_block {
public:
    statement_block(istr sb, istr se) :
        cl_block(sb, se, blk_type_expr) { }

    enum statement_type {
        for_statement,
        if_statement,
        endfor_statement,
        endif_statement,
    };

    bool is_begin_tag() {
        return stat_type_ == for_statement ||
            stat_type_ == if_statement;
    }

    bool is_end_tag() {
        return stat_type_ == endfor_statement ||
            stat_type_ == endif_statement;
    }

    bool matches(const statement_block& rhs) {
        if(rhs.stat_type_ == endfor_statement)
            return stat_type_ == for_statement;
        if(rhs.stat_type_ == endif_statement)
            return stat_type_ == if_statement;

        return false;
    }

private:
    statement_type stat_type_;
};

class cl_template {
public:
    cl_template(const std::string& src = "") :
        src_(src) { }

private:
    std::string src_;

    std::vector<std::unique_ptr<cl_block>> blocks_;

    void parse_(const std::string::const_iterator& it_beg,
            const std::string::const_iterator& it_end) {
        blocks_.clear();

        std::regex blk_pattern(R"EOF(\{(%\s*(.*?)\s*%|\{\s*(.*?)\s*\})\})EOF");
        std::smatch m;
        std::string::const_iterator it = it_beg;
        std::vector<statement_block*> tag_stack;

        while(true) {
            std::regex_search(it, it_end, m, blk_pattern);
            if(m.empty()) break;

            if(it != m[0].first)
                blocks_.emplace_back(new text_block(it, m[0].first));

            if(*(m[1].first) == '%') {
                statement_block* s;
                blocks_.emplace_back(s = new statement_block(
                        m[2].first, m[2].second));

                if(s->is_begin_tag())
                    tag_stack.push_back(s);
                else if(s->is_end_tag()) {
                    if(tag_stack.empty() || !tag_stack.back()->matches(*s))
                        throw parse_error("Failed on matching", m[2].first);
                    else tag_stack.pop_back();
                }
            }
            else if(*(m[1].first) == '{')
                blocks_.emplace_back(new expression_block(
                        m[3].first, m[3].second));

            it = m[0].second;
        }

        if(it != it_end)
            blocks_.emplace_back(new text_block(it, it_end));
    }
};

}

}

#endif // CLTMPL_H_INCLUDED
