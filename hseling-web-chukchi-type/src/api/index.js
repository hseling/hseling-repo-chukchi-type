import axios from "axios"

const API_URL = process.env.HSELING_API_ENDPOINT

export async function fetchSuggestions(text) {
    return axios.post(API_URL + "get_suggestions", {text: text})
}